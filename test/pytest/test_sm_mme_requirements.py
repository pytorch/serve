import os
import pathlib

import pytest
import requests
import test_utils
import torch

CURR_FILE_PATH = os.path.dirname(os.path.realpath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(CURR_FILE_PATH, "..", ".."))
MODELSTORE_DIR = os.path.join(REPO_ROOT, "model_store")
data_file_kitten = os.path.join(REPO_ROOT, "examples/image_classifier/kitten.jpg")
HF_TRANSFORMERS_EXAMPLE_DIR = os.path.join(
    REPO_ROOT, "examples/Huggingface_Transformers/"
)


def test_no_model_loaded():
    """
    Validates that TorchServe returns reponse code 404 if no model is loaded.
    """

    os.makedirs(MODELSTORE_DIR, exist_ok=True)  # Create modelstore directory
    test_utils.start_torchserve(model_store=MODELSTORE_DIR)

    response = requests.post(
        url="http://localhost:8080/models/alexnet/invoke",
        data=open(data_file_kitten, "rb"),
    )
    assert response.status_code == 404, "Model not loaded error expected"


@pytest.mark.skipif(
    not ((torch.cuda.device_count() > 0) and torch.cuda.is_available()),
    reason="Test to be run on GPU only",
)
def test_oom_on_model_load():
    """
    Validates that TorchServe returns reponse code 507 if there is OOM on model loading.
    """

    # Create model store directory
    pathlib.Path(test_utils.MODEL_STORE).mkdir(parents=True, exist_ok=True)

    # Start TorchServe
    test_utils.start_torchserve(no_config_snapshots=True)

    # Register model
    params = {
        "model_name": "BERTSeqClassification",
        "url": "https://torchserve.pytorch.org/mar_files/BERTSeqClassification.mar",
        "batch_size": 1,
        "initial_workers": 16,
    }
    response = test_utils.register_model_with_params(params)

    assert response.status_code == 507, "OOM Error expected"

    test_utils.stop_torchserve()


@pytest.mark.skipif(
    not ((torch.cuda.device_count() > 0) and torch.cuda.is_available()),
    reason="Test to be run on GPU only",
)
def test_oom_on_invoke():
    # Create model store directory
    pathlib.Path(test_utils.MODEL_STORE).mkdir(parents=True, exist_ok=True)

    # Start TorchServe
    test_utils.start_torchserve(no_config_snapshots=True)

    # Register model
    params = {
        "model_name": "BERTSeqClassification",
        "url": "https://torchserve.pytorch.org/mar_files/BERTSeqClassification.mar",
        "batch_size": 8,
        "initial_workers": 12,
    }
    response = test_utils.register_model_with_params(params)

    input_text = os.path.join(
        REPO_ROOT,
        "examples",
        "Huggingface_Transformers",
        "Seq_classification_artifacts",
        "sample_text_captum_input.txt",
    )

    # Make 8 curl requests in parallel with &
    # Send multiple requests to make sure to hit OOM
    for i in range(10):
        response = os.popen(
            f"curl http://127.0.0.1:8080/models/BERTSeqClassification/invoke -T {input_text} && "
            f"curl http://127.0.0.1:8080/models/BERTSeqClassification/invoke -T {input_text} && "
            f"curl http://127.0.0.1:8080/models/BERTSeqClassification/invoke -T {input_text} && "
            f"curl http://127.0.0.1:8080/models/BERTSeqClassification/invoke -T {input_text} && "
            f"curl http://127.0.0.1:8080/models/BERTSeqClassification/invoke -T {input_text} && "
            f"curl http://127.0.0.1:8080/models/BERTSeqClassification/invoke -T {input_text} && "
            f"curl http://127.0.0.1:8080/models/BERTSeqClassification/invoke -T {input_text} && "
            f"curl http://127.0.0.1:8080/models/BERTSeqClassification/invoke -T {input_text} "
        )
        response = response.read()

    # If OOM is hit, we expect code 507 to be present in the response string
    lines = response.split("\n")
    output = ""
    for line in lines:
        if "code" in line:
            line = line.strip()
            output = line
            break
    assert output == '"code": 507,', "OOM Error expected"
