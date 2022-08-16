import importlib
import json
import os
import shutil
import sys
import time
from argparse import Namespace
from pathlib import Path

import pytest
import requests
import test_utils
import torch
from test_utils import REPO_ROOT

from ts.torch_handler.unit_tests.test_utils.mock_context import MockContext

EXAMPLE_ROOT_DIR = os.path.join(
    REPO_ROOT, "examples", "text_classification_with_scriptable_tokenizer"
)

EXPECTED_RESULT_EMPTY_STRING = [
    {
        "0": pytest.approx(0.5625126361846924, 1e-3),
        "1": pytest.approx(0.43748739361763, 1e-3),
    }
]

EXPECTED_RESULT_SAMPLE_TEXT = [
    {
        "0": pytest.approx(0.5138629078865051, 1e-3),
        "1": pytest.approx(0.4861370921134949, 1e-3),
    }
]


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

    # Would be more elegant to mock RobertaEncoderConf and default num_encoder_layers to 1 but I failed do far
    from torchtext.models import RobertaClassificationHead
    from torchtext.models.roberta.bundler import (
        _TEXT_BUCKET,
        RobertaEncoderConf,
        T,
        load_state_dict_from_url,
        urljoin,
    )

    try:
        import torchtext.models.roberta.bundler.RobertaModelBundle as RobertaBundle
    except ImportError:
        from torchtext.models.roberta.bundler import RobertaBundle

    torch.manual_seed(42)

    XLMR_BASE_ENCODER = RobertaBundle(
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


@pytest.fixture(scope="module", name="jit_file_path")
def script_and_export_model(model, script_tokenizer_and_model, work_dir):
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


@pytest.fixture(scope="module", name="mar_file_path")
def create_mar_file(work_dir, session_mocker, jit_file_path, model_archiver):
    """
    Create mar file and return file path.
    """
    model_name = "scriptable_tokenizer_untrained"

    mar_file_path = os.path.join(work_dir, model_name + ".mar")

    args = Namespace(
        model_name=model_name,
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

    model_archiver.generate_model_archive()

    assert os.path.exists(mar_file_path)

    yield mar_file_path

    # Clean up files
    try:
        os.remove(mar_file_path)
    except OSError:
        pass


# Registering the module needs to be function scope until https://github.com/pytorch/text/issues/1849 is resolved
@pytest.fixture(scope="function", name="model_name")
def register_model(mar_file_path, model_store, torchserve):
    shutil.copy(mar_file_path, model_store)

    file_name = os.path.split(mar_file_path)[-1]

    model_name = os.path.splitext(file_name)[0]

    test_utils.reg_resp = test_utils.register_model(model_name, file_name)

    yield model_name

    test_utils.unregister_model(model_name)


@pytest.fixture
def test_file():
    return os.path.join(EXAMPLE_ROOT_DIR, "sample_text.txt")


def test_handler(monkeypatch, mocker, jit_file_path, test_file):
    monkeypatch.syspath_prepend(EXAMPLE_ROOT_DIR)

    # We need to recreate the handler to avoid running into https://github.com/pytorch/text/issues/1849
    def create_and_call_handler(input_text):

        from handler import CustomTextClassifier

        handler = CustomTextClassifier()
        ctx = MockContext(
            model_pt_file=Path(jit_file_path).name,
            model_dir=Path(jit_file_path).parent,
            model_file=None,
        )

        torch.manual_seed(42)
        handler.initialize(ctx)

        # Try empty string
        x = mocker.Mock(get=lambda _: input_text)

        x = handler.preprocess([x])
        x = handler.inference(x)
        x = handler.postprocess(x)
        return x

    res = create_and_call_handler("")

    assert res == EXPECTED_RESULT_EMPTY_STRING

    # Try sample text
    with open(test_file, "rb") as f:
        sample_text = f.readline()

    res = create_and_call_handler(sample_text)

    assert res == EXPECTED_RESULT_SAMPLE_TEXT


def test_inference_with_untrained_model_and_sample_text(model_name, test_file):

    with open(test_file, "rb") as f:
        response = requests.post(
            url=f"http://localhost:8080/predictions/{model_name}", data=f
        )

    assert response.status_code == 200

    result_entries = json.loads(response.text)

    assert "Negative" in result_entries
    assert "Positive" in result_entries

    # We're using an untrained model for the unit test, so results do not make sense but should be consistent
    assert float(result_entries["Negative"]) == EXPECTED_RESULT_SAMPLE_TEXT[0]["0"]
    assert float(result_entries["Positive"]) == EXPECTED_RESULT_SAMPLE_TEXT[0]["1"]


def test_inference_with_untrained_model_and_empty_string(model_name):

    data = "".encode("utf8")

    response = requests.post(
        url=f"http://localhost:8080/predictions/{model_name}", data=data
    )

    assert response.status_code == 200

    result_entries = json.loads(response.text)

    assert "Negative" in result_entries
    assert "Positive" in result_entries

    # We're using an untrained model for the unit test, so results do not make sense but should be consistent
    assert float(result_entries["Negative"]) == EXPECTED_RESULT_EMPTY_STRING[0]["0"]
    assert float(result_entries["Positive"]) == EXPECTED_RESULT_EMPTY_STRING[0]["1"]


def test_inference_with_pretrained_model(model_store, test_file, torchserve):
    model_name = "scriptable_tokenizer"

    params = (
        ("model_name", model_name),
        (
            "url",
            "https://bert-mar-file.s3.us-west-2.amazonaws.com/text_classification_with_scriptable_tokenizer/scriptable_tokenizer.mar",
        ),
        ("initial_workers", "1"),
    )

    test_utils.reg_resp = test_utils.register_model_with_params(params)

    # Give test some time for model to be downloaded from S3 bucket
    for sleep_time in [2, 4, 8, 16, 32, 64]:
        with open(test_file, "rb") as f:
            response = requests.post(
                url=f"http://localhost:8080/predictions/{model_name}", data=f
            )
        if response.status_code == 200:
            break
        time.sleep(sleep_time)

    assert response.status_code == 200

    result_entries = json.loads(response.text)

    assert "Negative" in result_entries
    assert "Positive" in result_entries

    assert float(result_entries["Negative"]) == pytest.approx(
        0.0001851904089562595, 1e-3
    )
    assert float(result_entries["Positive"]) == pytest.approx(0.9998148083686829, 1e-3)

    test_utils.unregister_model(model_name)
