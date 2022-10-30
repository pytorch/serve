"""
Unit test for near real-time video example
"""
import json
import os
import shutil
from argparse import Namespace
from pathlib import Path

import pytest
import requests
import test_utils
import torch

from ts.torch_handler.image_classifier import ImageClassifier
from ts.torch_handler.unit_tests.test_utils.mock_context import MockContext

CURR_FILE_PATH = Path(__file__).parent
REPO_ROOT_DIR = CURR_FILE_PATH.parent.parent
EXAMPLE_ROOT_DIR = REPO_ROOT_DIR.joinpath(
    "examples", "image_classifier", "near_real_time_video"
)
MODEL_PTH_FILE = "resnet18-f37072fd.pth"


# The test cases and expected results used for the unittests with batch size one
EXPECTED_RESULTS = [[["tabby", "tiger_cat", "Egyptian_cat", "lynx", "bucket"]]]
TEST_CASES = [
    ("kitten.jpg", EXPECTED_RESULTS[0]),
]

# Adding this as we can't pass extra files to MockContext
EXPECTED_RESULTS1 = [["281", "282", "285", "287", "463"]]
TEST_CASES1 = [
    ("kitten.jpg", EXPECTED_RESULTS1[0]),
]


@pytest.fixture(scope="module")
def work_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("work_dir")


@pytest.fixture(scope="module", name="mar_file_path")
def create_mar_file(work_dir, session_mocker, model_archiver):
    """
    Create mar file and return file path.
    """
    model_name = "resnet-18"

    mar_file_path = Path(work_dir).joinpath(model_name + ".mar")

    args = Namespace(
        model_name=model_name,
        version="1.0",
        serialized_file=os.path.join(REPO_ROOT_DIR, MODEL_PTH_FILE),
        model_file=EXAMPLE_ROOT_DIR.joinpath("model.py").as_posix(),
        handler="image_classifier",
        extra_files=REPO_ROOT_DIR.joinpath(
            "examples", "image_classifier", "index_to_name.json"
        ).as_posix(),
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

    assert mar_file_path.exists()

    yield mar_file_path.as_posix()

    # Clean up files
    mar_file_path.unlink(missing_ok=True)


@pytest.fixture(scope="module", name="model_name")
def register_model(mar_file_path, model_store, torchserve):
    """
    Register the model in torchserve
    """
    shutil.copy(mar_file_path, model_store)

    file_name = Path(mar_file_path).name

    model_name = Path(file_name).stem

    test_utils.reg_resp = test_utils.register_model(model_name, file_name)

    yield model_name

    test_utils.unregister_model(model_name)


@pytest.mark.parametrize(("file", "expected_result"), TEST_CASES1)
def test_handler(monkeypatch, mocker, file, expected_result):
    """
    Test dlrm handler as standalone entity with specified test cases
    """
    monkeypatch.syspath_prepend(EXAMPLE_ROOT_DIR)

    handler = ImageClassifier()
    ctx = MockContext(
        model_pt_file=REPO_ROOT_DIR.joinpath(MODEL_PTH_FILE).as_posix(),
        model_dir=EXAMPLE_ROOT_DIR.as_posix(),
        model_file="model.py",
    )

    torch.manual_seed(42 * 42)
    handler.initialize(ctx)
    data = {}
    # Batch size 2
    with open(Path(CURR_FILE_PATH) / "test_data" / file, "rb") as f:
        data["data"] = f
        data = f.read()

    x = mocker.Mock(get=lambda x: data)

    x = handler.preprocess([x])
    x = handler.inference(x)
    x = handler.postprocess(x)
    labels = list(x[0].keys())

    assert labels == expected_result


@pytest.mark.parametrize(("file", "expected_result"), TEST_CASES)
def test_inference_with_model_post_as_rb(model_name, file, expected_result):
    """
    Full circle test with torchserve
    """

    with open(Path(CURR_FILE_PATH) / "test_data" / file, "rb") as f:
        response = requests.post(
            url=f"http://localhost:8080/predictions/{model_name}", data=f
        )

    assert response.status_code == 200

    result_entries = json.loads(response.text)

    labels = list(result_entries.keys())

    assert [labels] == expected_result
