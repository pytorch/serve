"""
Tests for pytorch profiler integration
"""
# pylint: disable=W0613, W0621
import glob
import json
import os
import pathlib
import platform
import shutil
import subprocess
from concurrent import futures

import pytest
import requests
import test_utils

REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../")
)
data_file_mnist = os.path.join(
    REPO_ROOT, "examples", "image_classifier", "mnist", "test_data", "1.png"
)
data_file_resnet = os.path.join(
    REPO_ROOT,
    "examples",
    "image_classifier",
    "resnet_152_batch",
    "images",
    "kitten.jpg",
)
data_file_resnet_dog = os.path.join(
    REPO_ROOT, "examples", "image_classifier", "resnet_152_batch", "images", "dog.jpg"
)
profiler_utils = os.path.join(REPO_ROOT, "test", "pytest", "profiler_utils")

TF_INFERENCE_API = "http://127.0.0.1:8080"
TF_MANAGEMENT_API = "http://127.0.0.1:8081"
DEFAULT_OUTPUT_DIR = "/tmp/pytorch_profiler/resnet-152-batch"


@pytest.fixture
@pytest.mark.skipif(
    platform.machine() == "aarch64", reason="Test skipped on aarch64 architecture"
)
def set_custom_handler(handler_name):
    """
    This method downloads resnet serialized file, creates mar file and sets up a custom handler
    for running tests
    """
    os.environ["ENABLE_TORCH_PROFILER"] = "true"

    if os.path.exists(DEFAULT_OUTPUT_DIR):
        shutil.rmtree(DEFAULT_OUTPUT_DIR)

    ## Download resnet 152 serialized file
    pathlib.Path(test_utils.MODEL_STORE).mkdir(parents=True, exist_ok=True)
    serialized_file = os.path.join(test_utils.MODEL_STORE, "resnet152-394f9c45.pth")
    if not os.path.exists(serialized_file):
        response = requests.get(
            "https://download.pytorch.org/models/resnet152-394f9c45.pth",
            allow_redirects=True,
        )
        assert response.status_code == 200
        with open(serialized_file, "wb") as f:
            f.write(response.content)

    ## Generate mar file
    cmd = test_utils.model_archiver_command_builder(
        model_name="resnet-152-batch",
        version="1.0",
        model_file=os.path.join(
            test_utils.CODEBUILD_WD,
            "examples",
            "image_classifier",
            "resnet_152_batch",
            "model.py",
        ),
        serialized_file=serialized_file,
        handler=handler_name,
        extra_files=os.path.join(
            test_utils.CODEBUILD_WD,
            "examples",
            "image_classifier",
            "index_to_name.json",
        ),
        force=True,
    )
    print(cmd)
    cmd = cmd.split(" ")

    subprocess.run(cmd, check=True, timeout=1000)

    # Create config properties to enable env vars
    config_properties = os.path.join(test_utils.MODEL_STORE, "config.properties")
    with open(config_properties, "w") as fp:
        fp.write("enable_envvars_config=true")

    test_utils.start_torchserve(
        no_config_snapshots=True, gen_mar=False, snapshot_file=config_properties
    )

    # Register resnet model
    params = {
        "model_name": "resnet152",
        "url": "resnet-152-batch.mar",
        "batch_size": 4,
        "max_batch_delay": 5000,
        "initial_workers": 3,
        "synchronous": "true",
    }
    test_utils.register_model_with_params(params)


@pytest.mark.parametrize(
    "handler_name",
    [os.path.join(profiler_utils, "resnet_custom.py"), "image_classifier"],
)
@pytest.mark.skipif(
    platform.machine() == "aarch64", reason="Test skipped on aarch64 architecture"
)
def test_profiler_default_and_custom_handler(set_custom_handler, handler_name):
    """
    Tests pytorch profiler integration with default and custom handler
    """
    assert os.path.exists(data_file_resnet)
    data = open(data_file_resnet, "rb")
    response = requests.post("{}/predictions/resnet152".format(TF_INFERENCE_API), data)
    assert "tiger_cat" in json.loads(response.content)
    assert len(glob.glob("{}/*.pt.trace.json".format(DEFAULT_OUTPUT_DIR))) == 1
    test_utils.unregister_model("resnet152")
    shutil.rmtree(DEFAULT_OUTPUT_DIR)
    test_utils.torchserve_cleanup()


@pytest.mark.parametrize(
    "handler_name",
    [os.path.join(profiler_utils, "resnet_profiler_override.py")],
)
@pytest.mark.skipif(
    platform.machine() == "aarch64", reason="Test skipped on aarch64 architecture"
)
def test_profiler_arguments_override(set_custom_handler, handler_name):
    """
    Tests pytorch profiler integration when user overrides the profiler arguments
    """
    CUSTOM_PATH = "/tmp/output/resnet-152-batch"
    if os.path.exists(CUSTOM_PATH):
        shutil.rmtree(CUSTOM_PATH)
    assert os.path.exists(data_file_resnet)
    data = open(data_file_resnet, "rb")
    response = requests.post("{}/predictions/resnet152".format(TF_INFERENCE_API), data)
    assert "tiger_cat" in json.loads(response.content)
    assert len(glob.glob("{}/*.pt.trace.json".format(CUSTOM_PATH))) == 1
    test_utils.unregister_model("resnet152")
    shutil.rmtree(CUSTOM_PATH)
    test_utils.torchserve_cleanup()


@pytest.mark.parametrize(
    "handler_name",
    [os.path.join(profiler_utils, "resnet_profiler_override.py")],
)
@pytest.mark.skipif(
    platform.machine() == "aarch64", reason="Test skipped on aarch64 architecture"
)
def test_batch_input(set_custom_handler, handler_name):
    """
    Tests pytorch profiler integration with batch inference
    """

    CUSTOM_PATH = "/tmp/output/resnet-152-batch"

    if os.path.exists(CUSTOM_PATH):
        shutil.rmtree(CUSTOM_PATH)
    assert os.path.exists(data_file_resnet)

    def invoke_batch_input():
        data = open(data_file_resnet, "rb")
        response = requests.post(
            "{}/predictions/resnet152".format(TF_INFERENCE_API), data
        )
        assert response.status_code == 200
        assert "tiger_cat" in json.loads(response.content)

    with futures.ThreadPoolExecutor(2) as executor:
        for _ in range(2):
            executor.submit(invoke_batch_input)

    assert len(glob.glob("{}/*.pt.trace.json".format(CUSTOM_PATH))) == 1
    test_utils.unregister_model("resnet152")
    shutil.rmtree(CUSTOM_PATH)
    test_utils.torchserve_cleanup()
