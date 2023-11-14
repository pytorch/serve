import json
import os
import subprocess
from shutil import copy

import pytest
import requests
import test_utils
import torch

CURR_FILE_PATH = os.path.dirname(os.path.realpath(__file__))
REPO_ROOT_DIR = os.path.normpath(os.path.join(CURR_FILE_PATH, "..", ".."))
MODEL_STORE_DIR = os.path.join(REPO_ROOT_DIR, "model_store")
TORCH_TENSORRT_EXAMPLE_DIR = os.path.join(REPO_ROOT_DIR, "examples", "torch_tensorrt")
TORCH_TENSORRT_MAR_FILE = os.path.join(REPO_ROOT_DIR, "res50-trt-fp16.mar")
EXPECTED_RESULTS = ["tabby", "tiger_cat", "Egyptian_cat", "lynx", "lens_cap"]

tensorrt_available = False
cmd = ["python", "-c", "import tensorrt"]
r = subprocess.run(cmd)
tensorrt_available = r.returncode == 0

torch_tensorrt_available = False
cmd = ["python", "-c", "import torch_tensorrt"]
r = subprocess.run(cmd)
torch_tensorrt_available = r.returncode == 0

tensorrt_and_torch_tensorrt_available = tensorrt_available and torch_tensorrt_available


def setup_module():
    test_utils.torchserve_cleanup()
    create_example_mar()

    os.makedirs(MODEL_STORE_DIR, exist_ok=True)
    copy(TORCH_TENSORRT_MAR_FILE, MODEL_STORE_DIR)

    test_utils.start_torchserve(model_store=MODEL_STORE_DIR)


def teardown_module():
    test_utils.torchserve_cleanup()

    test_utils.delete_model_store(MODEL_STORE_DIR)
    os.rmdir(MODEL_STORE_DIR)

    delete_example_mar()


def create_example_mar():
    if not os.path.exists(TORCH_TENSORRT_MAR_FILE):
        create_serialized_file_cmd = f"cd {REPO_ROOT_DIR};python {os.path.join(TORCH_TENSORRT_EXAMPLE_DIR, 'resnet_tensorrt.py')}"
        subprocess.check_call(create_serialized_file_cmd, shell=True)
        create_mar_cmd = (
            f"torch-model-archiver --model-name res50-trt-fp16 --handler image_classifier --version 1.0 --serialized-file res50_trt_fp16.pt --extra-files "
            f"{os.path.join(REPO_ROOT_DIR, 'examples', 'image_classifier', 'index_to_name.json')}"
        )
        subprocess.check_call(create_mar_cmd, shell=True)


def delete_example_mar():
    try:
        os.remove(TORCH_TENSORRT_MAR_FILE)
    except OSError:
        pass


@pytest.mark.skipif(
    not (tensorrt_and_torch_tensorrt_available and torch.cuda.is_available()),
    reason="Make sure tensorrt and torch-tensorrt are installed and torch.cuda is available",
)
def test_model_archive_creation():
    assert os.path.exists(
        TORCH_TENSORRT_MAR_FILE
    ), "Failed to create torch tensorrt mar file"


@pytest.mark.skipif(
    not (tensorrt_and_torch_tensorrt_available and torch.cuda.is_available()),
    reason="Make sure tensorrt and torch-tensorrt are installed and torch.cuda is available",
)
def test_model_register_unregister():
    reg_resp = test_utils.register_model("res50-trt-fp16", "res50-trt-fp16.mar")
    assert reg_resp.status_code == 200, "Model Registration Failed"

    unreg_resp = test_utils.unregister_model("res50-trt-fp16")
    assert unreg_resp.status_code == 200, "Model Unregistration Failed"


@pytest.mark.skipif(
    not (tensorrt_and_torch_tensorrt_available and torch.cuda.is_available()),
    reason="Make sure tensorrt and torch-tensorrt are installed and torch.cuda is available",
)
def test_run_inference_torch_tensorrt():
    test_utils.register_model("res50-trt-fp16", "res50-trt-fp16.mar")
    image_path = os.path.join(REPO_ROOT_DIR, "examples/image_classifier/kitten.jpg")
    with open(image_path, "rb") as file:
        image_data = file.read()
    payload = {"data": image_data}
    response = requests.post(
        url="http://localhost:8080/predictions/res50-trt-fp16", files=payload
    )
    assert response.status_code == 200, "Image prediction failed"
    result_dict = json.loads(response.content.decode("utf-8"))
    labels = list(result_dict.keys())
    assert labels == EXPECTED_RESULTS, "Image prediction labels do not match"
    test_utils.unregister_model("res50-trt-fp16")
