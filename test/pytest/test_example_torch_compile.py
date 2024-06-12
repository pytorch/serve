import os
import sys
from pathlib import Path

import pytest
import torch
from pkg_resources import packaging

from ts.torch_handler.image_classifier import ImageClassifier
from ts.torch_handler.unit_tests.test_utils.mock_context import MockContext
from ts.utils.util import load_label_mapping
from ts_scripts.utils import try_and_handle

CURR_FILE_PATH = Path(__file__).parent.absolute()
REPO_ROOT_DIR = CURR_FILE_PATH.parents[1]
EXAMPLE_ROOT_DIR = REPO_ROOT_DIR.joinpath("examples", "pt2", "torch_compile")
TEST_DATA = REPO_ROOT_DIR.joinpath("examples", "image_classifier", "kitten.jpg")
MAPPING_DATA = REPO_ROOT_DIR.joinpath(
    "examples", "image_classifier", "index_to_name.json"
)
MODEL_PTH_FILE = "densenet161-8d451a50.pth"
MODEL_FILE = "model.py"
MODEL_YAML_CFG_FILE = EXAMPLE_ROOT_DIR.joinpath("model-config.yaml")


PT2_AVAILABLE = (
    True
    if packaging.version.parse(torch.__version__) > packaging.version.parse("2.0")
    else False
)

EXPECTED_RESULTS = ["tabby", "tiger_cat", "Egyptian_cat", "lynx", "plastic_bag"]


@pytest.fixture(scope="function")
def chdir_example(monkeypatch):
    # Change directory to example directory
    monkeypatch.chdir(EXAMPLE_ROOT_DIR)
    monkeypatch.syspath_prepend(EXAMPLE_ROOT_DIR)
    yield

    # Teardown
    monkeypatch.undo()

    # Delete imported model
    model = MODEL_FILE.split(".")[0]
    if model in sys.modules:
        del sys.modules[model]


@pytest.mark.skipif(PT2_AVAILABLE == False, reason="torch version is < 2.0")
def test_torch_compile_inference(chdir_example):
    # Download weights
    if not os.path.isfile(EXAMPLE_ROOT_DIR.joinpath(MODEL_PTH_FILE)):
        try_and_handle(
            f"wget https://download.pytorch.org/models/{MODEL_PTH_FILE} -P {EXAMPLE_ROOT_DIR}"
        )

    # Handler for Image classification
    handler = ImageClassifier()

    # Context definition
    ctx = MockContext(
        model_pt_file=MODEL_PTH_FILE,
        model_dir=EXAMPLE_ROOT_DIR.as_posix(),
        model_file=MODEL_FILE,
        model_yaml_config_file=MODEL_YAML_CFG_FILE,
    )

    torch.manual_seed(42 * 42)
    handler.initialize(ctx)
    handler.context = ctx
    handler.mapping = load_label_mapping(MAPPING_DATA)

    data = {}
    with open(TEST_DATA, "rb") as image:
        image_file = image.read()
        byte_array_type = bytearray(image_file)
        data["body"] = byte_array_type

    result = handler.handle([data], ctx)

    labels = list(result[0].keys())

    assert labels == EXPECTED_RESULTS
