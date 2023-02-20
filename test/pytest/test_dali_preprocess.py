"""
Unit test for near real-time video example
"""
import os
from pathlib import Path

import pytest
import requests

from examples.nvidia_dali.custom_handler import DALIHandler
from ts.torch_handler.unit_tests.test_utils.mock_context import MockContext

CURR_FILE_PATH = Path(__file__).parent
REPO_ROOT_DIR = CURR_FILE_PATH.parent.parent


EXAMPLE_ROOT_DIR_DALI = REPO_ROOT_DIR.joinpath("examples", "nvidia_dali")

MODEL_PTH_FILE = "resnet18-f37072fd.pth"

EXAMPLE_ROOT_DIR_RESNET = REPO_ROOT_DIR.joinpath(
    "examples", "image_classifier", "resnet_18"
)

EXPECTED_RESULTS = [["tabby", "tiger_cat", "Egyptian_cat", "lynx", "bucket"]]
TEST_CASES = [
    ("kitten.jpg", EXPECTED_RESULTS[0]),
]


@pytest.mark.parametrize(("file", "expected_result"), TEST_CASES)
def test_dali_preprocess(monkeypatch, file, expected_result):

    monkeypatch.syspath_prepend(EXAMPLE_ROOT_DIR_RESNET)

    serialized_file = os.path.join(REPO_ROOT_DIR, MODEL_PTH_FILE)
    if not os.path.exists(serialized_file):
        response = requests.get(
            "https://download.pytorch.org/models/resnet18-f37072fd.pth",
            allow_redirects=True,
        )
        assert response.status_code == 200
        with open(serialized_file, "wb") as f:
            f.write(response.content)

    handler = DALIHandler()
    ctx = MockContext(
        model_pt_file=REPO_ROOT_DIR.joinpath(MODEL_PTH_FILE).as_posix(),
        model_dir=EXAMPLE_ROOT_DIR_DALI,
        model_file="model.py",
    )

    handler.initialize(ctx)
    data = {}

    with open(Path(CURR_FILE_PATH) / "test_data" / file, "rb") as image:
        image_file = image.read()
        byte_array_type = bytearray(image_file)
        data["body"] = byte_array_type

    x = handler.preprocess([data])
    x = handler.inference(x)
    x = handler.postprocess(x)
    labels = list(x[0].keys())

    assert labels == expected_result
