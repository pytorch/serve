import os
import tempfile
from pathlib import Path

import pytest
import requests
import test_utils

ROOT_DIR = os.path.join(tempfile.gettempdir(), "workspace")
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

expected_output = {
    "code": 405,
    "type": "MethodNotAllowedException",
    "message": "Requested method is not allowed, please refer to API document.",
}


@pytest.fixture(scope="module")
def setup_torchserve():
    MODEL_STORE = os.path.join(ROOT_DIR, "model_store/")

    Path(test_utils.MODEL_STORE).mkdir(parents=True, exist_ok=True)

    test_utils.start_torchserve(
        no_config_snapshots=True, models="mnist=mnist.mar", model_api_enabled=False
    )

    yield "test"

    test_utils.stop_torchserve()


@pytest.fixture(scope="module")
def setup_torchserve_api_enabled():
    MODEL_STORE = os.path.join(ROOT_DIR, "model_store/")

    Path(test_utils.MODEL_STORE).mkdir(parents=True, exist_ok=True)

    test_utils.start_torchserve(no_config_snapshots=True, models="mnist=mnist.mar")

    yield "test"

    test_utils.stop_torchserve()


# Test register a model after startup - Model control mode: default
def test_register_model_failing(setup_torchserve):
    response = requests.get("http://localhost:8081/models/mnist")
    assert response.status_code == 200, "management check failed"
    params = (
        ("model_name", "resnet-18"),
        ("url", "resnet-18.mar"),
        ("initial_workers", "1"),
        ("synchronous", "true"),
    )
    response = requests.post("http://localhost:8081/models", params=params)

    assert response.status_code == 405, "model control check failed"
    assert response.json() == expected_output, "unexpected exception"
    response = requests.get("http://localhost:8081/models/resnet-18")
    assert response.status_code == 404, "management check failed"


# Test deleting a model after startup - Model control mode: default
def test_delete_model_failing(setup_torchserve):
    response = requests.get("http://localhost:8081/models/mnist")
    assert response.status_code == 200, "management check failed"

    response = requests.delete("http://localhost:8081/models/mnist")

    assert response.status_code == 405, "model control check failed"
    assert response.json() == expected_output, "unexpected exception"
    response = requests.get("http://localhost:8081/models/mnist")
    assert response.status_code == 200, "management check failed"


# Test register a model after startup - Model control mode: api enabled
def test_register_model(setup_torchserve_api_enabled):
    response = requests.get("http://localhost:8081/models/mnist")
    assert response.status_code == 200, "management check failed"
    params = (
        ("model_name", "resnet-18"),
        ("url", "resnet-18.mar"),
        ("initial_workers", "1"),
        ("synchronous", "true"),
    )
    response = requests.post("http://localhost:8081/models", params=params)

    assert response.status_code == 200, "model control check failed"
    response = requests.get("http://localhost:8081/models/resnet-18")
    assert response.status_code == 200, "management check failed"


# Test delete a model after startup - Model control mode: api enabled
def test_delete_model(setup_torchserve_api_enabled):
    response = requests.get("http://localhost:8081/models/mnist")
    assert response.status_code == 200, "management check failed"

    response = requests.delete("http://localhost:8081/models/mnist")
    assert response.status_code == 200, "model control check failed"


# Test priority between config.properties and cmd
# config sets mode to default and cmd sets to api enabled
# Priority falls to cmd
def test_priority():
    MODEL_STORE = os.path.join(ROOT_DIR, "model_store/")
    Path(test_utils.MODEL_STORE).mkdir(parents=True, exist_ok=True)

    config_file_priority = os.path.join(
        REPO_ROOT, "../resources/config_model_mode.properties"
    )
    test_utils.start_torchserve(
        snapshot_file=config_file_priority,
        no_config_snapshots=True,
        models="mnist=mnist.mar",
    )

    response = requests.delete("http://localhost:8081/models/mnist")

    test_utils.stop_torchserve()

    assert response.status_code == 200, "model control check failed"
