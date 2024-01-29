from pathlib import Path

import torch
from pkg_resources import packaging

from ts.torch_handler.image_classifier import ImageClassifier
from ts.torch_handler.unit_tests.test_utils.mock_context import MockContext
from ts.utils.util import load_label_mapping
from ts_scripts.utils import try_and_handle

CURR_FILE_PATH = Path(__file__).parent.absolute()
REPO_ROOT_DIR = CURR_FILE_PATH.parents[1]
EXAMPLE_ROOT_DIR = REPO_ROOT_DIR.joinpath("examples", "pt2", "torch_export_aot_compile")
TEST_DATA = REPO_ROOT_DIR.joinpath("examples", "image_classifier", "kitten.jpg")
MAPPING_DATA = REPO_ROOT_DIR.joinpath(
    "examples", "image_classifier", "index_to_name.json"
)
MODEL_SO_FILE = "resnet18_pt2.so"
MODEL_YAML_CFG_FILE = EXAMPLE_ROOT_DIR.joinpath("model-config.yaml")


PT_230_AVAILABLE = (
    True
    if packaging.version.parse(torch.__version__) > packaging.version.parse("2.2.2")
    else False
)

EXPECTED_RESULTS = ["tabby", "tiger_cat", "Egyptian_cat", "lynx", "bucket"]
TEST_CASES = [
    ("kitten.jpg", EXPECTED_RESULTS[0]),
]

BATCH_SIZE = 32


import os

import pytest


@pytest.fixture
def custom_working_directory(tmp_path):
    # Set the custom working directory
    custom_dir = tmp_path / "model_dir"
    custom_dir.mkdir()
    os.chdir(custom_dir)
    yield custom_dir
    # Clean up and return to the original working directory
    os.chdir(tmp_path)


@pytest.mark.skipif(PT_230_AVAILABLE == False, reason="torch version is < 2.3.0")
def test_torch_export_aot_compile(custom_working_directory):
    # Get the path to the custom working directory
    model_dir = custom_working_directory

    # Construct the path to the Python script to execute
    script_path = os.path.join(EXAMPLE_ROOT_DIR, "resnet18_torch_export.py")

    # Get the .pt2 file from torch.export
    cmd = "python " + script_path
    try_and_handle(cmd)

    # Handler for Image classification
    handler = ImageClassifier()

    # Context definition
    ctx = MockContext(
        model_pt_file=MODEL_SO_FILE,
        model_dir=model_dir.as_posix(),
        model_file=None,
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


@pytest.mark.skipif(PT_230_AVAILABLE == False, reason="torch version is < 2.3.0")
def test_torch_export_aot_compile_dynamic_batching(custom_working_directory):
    # Get the path to the custom working directory
    model_dir = custom_working_directory

    # Construct the path to the Python script to execute
    script_path = os.path.join(EXAMPLE_ROOT_DIR, "resnet18_torch_export.py")

    # Get the .pt2 file from torch.export
    cmd = "python " + script_path
    try_and_handle(cmd)

    # Handler for Image classification
    handler = ImageClassifier()

    # Context definition
    ctx = MockContext(
        model_pt_file=MODEL_SO_FILE,
        model_dir=model_dir.as_posix(),
        model_file=None,
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

    # Send a batch of BATCH_SIZE elements
    result = handler.handle([data for i in range(BATCH_SIZE)], ctx)

    assert len(result) == BATCH_SIZE
