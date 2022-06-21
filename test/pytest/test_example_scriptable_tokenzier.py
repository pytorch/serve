import importlib
import json
import os
import shutil
import sys
from argparse import Namespace

import pytest
import requests
import test_utils
import torch
from torchtext.models import RobertaClassificationHead

CURR_FILE_PATH = os.path.dirname(os.path.realpath(__file__))
REPO_ROOT_DIR = os.path.normpath(os.path.join(CURR_FILE_PATH, "..", ".."))
EXAMPLE_ROOT_DIR = os.path.join(
    REPO_ROOT_DIR, "examples", "text_classification_with_scriptable_tokenizer"
)


@pytest.fixture(scope="module")
def work_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("work_dir")


@pytest.fixture(scope="module")
def model():
    """
    Rebuild XLMR model from training script but with reduces layer count to speed up unit test
    """
    num_classes = 2
    input_dim = 768
    """
    This reduces runtime by 3 seconds. Maybe not worth it.
    """
    # Would be more elegant to mock RobertaEncoderConf and default num_encoder_layers to 1 but I failed do far
    from torchtext.models.roberta.bundler import (
        _TEXT_BUCKET,
        RobertaEncoderConf,
        RobertaModelBundle,
        T,
        load_state_dict_from_url,
        urljoin,
    )

    torch.manual_seed(42)

    XLMR_BASE_ENCODER = RobertaModelBundle(
        _path=urljoin(_TEXT_BUCKET, "xlmr.base.encoder.pt"),
        _encoder_conf=RobertaEncoderConf(vocab_size=250002, num_encoder_layers=1),
        transform=lambda: T.Sequential(
            T.SentencePieceTokenizer(
                urljoin(_TEXT_BUCKET, "xlmr.sentencepiece.bpe.model")
            ),
            T.VocabTransform(
                load_state_dict_from_url(urljoin(_TEXT_BUCKET, "xlmr.vocab.pt"))
            ),
            T.Truncate(254),
            T.AddToken(token=0, begin=True),
            T.AddToken(token=2, begin=False),
        ),
    )

    classifier_head = RobertaClassificationHead(
        num_classes=num_classes, input_dim=input_dim
    )
    model = XLMR_BASE_ENCODER.get_model(head=classifier_head, load_weights=False)

    yield model


@pytest.fixture(scope="module")
def script_tokenizer_and_model(session_mocker, model):
    """
    This loads the source from script_tokenizer_and_model.py script and executes main
    We do this through import lib instead of just running the script to inject our smaller model
    """
    script_path = os.path.join(EXAMPLE_ROOT_DIR, "script_tokenizer_and_model.py")

    loader = importlib.machinery.SourceFileLoader(
        "script_tokenizer_and_model", script_path
    )
    spec = importlib.util.spec_from_loader("script_tokenizer_and_model", loader)
    script_tokenizer_and_model = importlib.util.module_from_spec(spec)

    sys.modules["script_tokenizer_and_model"] = script_tokenizer_and_model

    loader.exec_module(script_tokenizer_and_model)
    session_mocker.patch(
        "script_tokenizer_and_model.XLMR_BASE_ENCODER.get_model", return_value=model
    )

    yield script_tokenizer_and_model

    del sys.modules["script_tokenizer_and_model"]


@pytest.fixture(scope="module")
def jit_file_path(model, script_tokenizer_and_model, work_dir):
    """
    Create model and jit scripted model
    """
    # Define paths
    model_file_path = os.path.join(work_dir, "model.pt")
    jit_file_path = os.path.join(work_dir, "model_jit.pt")

    torch.save(model.state_dict(), model_file_path)

    script_tokenizer_and_model.main(
        Namespace(input_file=model_file_path, output_file=jit_file_path)
    )

    yield jit_file_path

    # Clean up files
    try:
        os.remove(model_file_path)
        os.remove(jit_file_path)
    except OSError:
        pass


@pytest.fixture(scope="module")
def archiver():
    loader = importlib.machinery.SourceFileLoader(
        "archiver",
        os.path.join(
            REPO_ROOT_DIR, "model-archiver", "model_archiver", "model_packaging.py"
        ),
    )
    spec = importlib.util.spec_from_loader("archiver", loader)
    archiver = importlib.util.module_from_spec(spec)

    sys.modules["archiver"] = archiver

    loader.exec_module(archiver)

    yield archiver

    del sys.modules["archiver"]


@pytest.fixture(scope="module")
def mar_file_path(work_dir, session_mocker, jit_file_path, archiver):
    """
    Create mar file and return file path.
    """
    mar_file_path = os.path.join(work_dir, "scriptable_tokenizer.mar")

    args = Namespace(
        model_name="scriptable_tokenizer",
        version="1.0",
        serialized_file=jit_file_path,
        model_file=None,
        handler=os.path.join(EXAMPLE_ROOT_DIR, "handler.py"),
        extra_files=os.path.join(EXAMPLE_ROOT_DIR, "index_to_name.json"),
        export_path=work_dir,
        requirements_file=None,
        runtime="python",
        force=False,
        archive_format="default",
    )

    mock = session_mocker.MagicMock()
    mock.parse_args = session_mocker.MagicMock(return_value=args)
    session_mocker.patch(
        "archiver.ArgParser.export_model_args_parser", return_value=mock
    )

    # Using ZIP_STORED instead of ZIP_DEFLATED reduces test runtime from 54 secs to 10 secs
    from zipfile import ZIP_STORED, ZipFile

    session_mocker.patch(
        "model_archiver.model_packaging_utils.zipfile.ZipFile",
        lambda x, y, _: ZipFile(x, y, ZIP_STORED),
    )

    archiver.generate_model_archive()

    assert os.path.exists(mar_file_path)

    yield mar_file_path

    # Clean up files
    try:
        os.remove(mar_file_path)
    except OSError:
        pass


@pytest.fixture(scope="module")
def model_store(work_dir):
    model_store_path = os.path.join(work_dir, "model_store")
    os.makedirs(model_store_path, exist_ok=True)

    yield model_store_path

    try:
        shutil.rmtree(model_store_path)
    except OSError:
        pass


@pytest.fixture(scope="module")
def torchserve(model_store):
    test_utils.torchserve_cleanup()

    test_utils.start_torchserve(
        model_store=model_store, no_config_snapshots=True, gen_mar=False
    )

    yield

    test_utils.torchserve_cleanup()


@pytest.fixture(scope="module")
def model_name(mar_file_path, model_store, torchserve):
    shutil.copy(mar_file_path, model_store)

    file_name = os.path.split(mar_file_path)[-1]

    model_name = os.path.splitext(file_name)[0]

    test_utils.reg_resp = test_utils.register_model(model_name, file_name)

    yield model_name

    test_utils.unregister_model(model_name)


def test_inference_with_sample_text(model_name):

    test_file = os.path.join(EXAMPLE_ROOT_DIR, "sample_text.txt")

    with open(test_file, "rb") as f:
        response = requests.post(
            url=f"http://localhost:8080/predictions/{model_name}", data=f
        )

    assert response.status_code == 200

    result_entries = json.loads(response.text)

    assert "Negative" in result_entries
    assert "Positive" in result_entries

    # We're using an untrained model for the unit test, so results do not make sense but should be consistent
    assert float(result_entries["Negative"]) == pytest.approx(0.6339209079742432, 1e-3)
    assert float(result_entries["Positive"]) == pytest.approx(0.36607906222343445, 1e-3)


def test_inference_with_empty_string(model_name):

    data = "".encode("utf8")

    response = requests.post(
        url=f"http://localhost:8080/predictions/{model_name}", data=data
    )

    assert response.status_code == 200

    result_entries = json.loads(response.text)

    assert "Negative" in result_entries
    assert "Positive" in result_entries

    # We're using an untrained model for the unit test, so results do not make sense but should be consistent
    assert float(result_entries["Negative"]) == pytest.approx(0.6082412600517273, 1e-3)
    assert float(result_entries["Positive"]) == pytest.approx(0.3917587101459503, 1e-3)
