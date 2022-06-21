import importlib
import os
from argparse import Namespace

import pytest
import torch
from torchtext.models import RobertaClassificationHead

CURR_FILE_PATH = os.path.dirname(os.path.realpath(__file__))
REPO_ROOT_DIR = os.path.normpath(os.path.join(CURR_FILE_PATH, "..", ".."))
EXAMPLE_ROOT_DIR = os.path.join(
    REPO_ROOT_DIR, "examples", "text_classification_with_scriptable_tokenizer"
)


@pytest.fixture
def model():
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


@pytest.fixture
def mar_file(model, tmp_path, mocker):
    model_file_path = os.path.join(tmp_path, "model.pt")
    jit_file_path = os.path.join(tmp_path, "model_jit.pt")
    mar_file_path = os.path.join(tmp_path, "scriptable_tokenizer.mar")

    script_path = os.path.join(EXAMPLE_ROOT_DIR, "script_tokenizer_and_model.py")

    torch.save(model.state_dict(), model_file_path)

    loader = importlib.machinery.SourceFileLoader(
        "script_tokenizer_and_model", script_path
    )
    spec = importlib.util.spec_from_loader("script_tokenizer_and_model", loader)
    script_tokenizer_and_model = importlib.util.module_from_spec(spec)

    import sys

    sys.modules["script_tokenizer_and_model"] = script_tokenizer_and_model

    loader.exec_module(script_tokenizer_and_model)
    mocker.patch(
        "script_tokenizer_and_model.XLMR_BASE_ENCODER.get_model", return_value=model
    )

    script_tokenizer_and_model.main(
        Namespace(input_file=model_file_path, output_file=jit_file_path)
    )

    loader = importlib.machinery.SourceFileLoader(
        "archiver",
        os.path.join(
            REPO_ROOT_DIR, "model-archiver", "model_archiver", "model_packaging.py"
        ),
    )
    spec = importlib.util.spec_from_loader("archiver", loader)
    archiver = importlib.util.module_from_spec(spec)

    import sys

    sys.modules["archiver"] = archiver

    args = Namespace(
        model_name="scriptable_tokenizer",
        version="1.0",
        serialized_file=jit_file_path,
        model_file=None,
        handler=os.path.join(EXAMPLE_ROOT_DIR, "handler.py"),
        extra_files=os.path.join(EXAMPLE_ROOT_DIR, "index_to_name.json"),
        export_path=tmp_path,
        requirements_file=None,
        runtime="python",
        force=False,
        archive_format="default",
    )

    loader.exec_module(archiver)
    mock = mocker.MagicMock()
    mock.parse_args = mocker.MagicMock(return_value=args)
    mocker.patch("archiver.ArgParser.export_model_args_parser", return_value=mock)

    # Using ZIP_STORED instead of ZIP_DEFLATED reduces test runtime by 44 secs
    from zipfile import ZIP_STORED, ZipFile

    mocker.patch(
        "model_archiver.model_packaging_utils.zipfile.ZipFile",
        lambda x, y, _: ZipFile(x, y, ZIP_STORED),
    )

    archiver.generate_model_archive()

    assert os.path.exists(mar_file_path)

    yield mar_file_path

    os.remove(model_file_path)
    os.remove(jit_file_path)
    os.remove(mar_file_path)


def test_inference(mar_file):
    pass
