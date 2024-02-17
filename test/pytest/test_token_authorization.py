import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import pytest
import requests
import test_utils

ROOT_DIR = os.path.join(tempfile.gettempdir(), "workspace")
REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../")
data_file_zero = os.path.join(REPO_ROOT, "test/pytest/test_data/0.png")
config_file = os.path.join(REPO_ROOT, "test/resources/config_token.properties")


# Set up token plugin
def get_plugin_jar():
    new_folder_path = os.path.join(ROOT_DIR, "plugins-path")
    plugin_folder = os.path.join(REPO_ROOT, "plugins")
    os.makedirs(new_folder_path, exist_ok=True)
    os.chdir(plugin_folder)
    subprocess.run(["./gradlew", "formatJava"])
    result = subprocess.run(["./gradlew", "build"])
    jar_path = os.path.join(plugin_folder, "endpoints/build/libs")
    jar_file = [file for file in os.listdir(jar_path) if file.endswith(".jar")]
    if jar_file:
        shutil.move(
            os.path.join(jar_path, jar_file[0]),
            os.path.join(new_folder_path, jar_file[0]),
        )
    os.chdir(REPO_ROOT)


# Parse json file and return key
def read_key_file(type):
    json_file_path = os.path.join(REPO_ROOT, "key_file.json")
    with open(json_file_path) as json_file:
        json_data = json.load(json_file)

    options = {
        "management": json_data.get("management", {}).get("key", "NOT_PRESENT"),
        "inference": json_data.get("inference", {}).get("key", "NOT_PRESENT"),
        "token": json_data.get("API", {}).get("key", "NOT_PRESENT"),
    }
    key = options.get(type, "Invalid data type")
    return key


@pytest.fixture(scope="module")
def setup_torchserve():
    get_plugin_jar()
    MODEL_STORE = os.path.join(ROOT_DIR, "model_store/")
    PLUGIN_STORE = os.path.join(ROOT_DIR, "plugins-path")

    Path(test_utils.MODEL_STORE).mkdir(parents=True, exist_ok=True)

    test_utils.start_torchserve(no_config_snapshots=True, plugin_folder=PLUGIN_STORE)

    key = read_key_file("management")
    header = {"Authorization": f"Bearer {key}"}

    params = (
        ("model_name", "mnist"),
        ("url", "mnist.mar"),
        ("initial_workers", "1"),
        ("synchronous", "true"),
    )
    response = requests.post(
        "http://localhost:8081/models", params=params, headers=header
    )
    file_content = Path(f"{REPO_ROOT}/key_file.json").read_text()
    print(file_content)

    yield "test"

    test_utils.stop_torchserve()


@pytest.fixture(scope="module")
def setup_torchserve_expiration():
    get_plugin_jar()
    MODEL_STORE = os.path.join(ROOT_DIR, "model_store/")
    PLUGIN_STORE = os.path.join(ROOT_DIR, "plugins-path")

    Path(test_utils.MODEL_STORE).mkdir(parents=True, exist_ok=True)

    test_utils.start_torchserve(
        snapshot_file=config_file, no_config_snapshots=True, plugin_folder=PLUGIN_STORE
    )

    key = read_key_file("management")
    header = {"Authorization": f"Bearer {key}"}

    params = (
        ("model_name", "mnist"),
        ("url", "mnist.mar"),
        ("initial_workers", "1"),
        ("synchronous", "true"),
    )
    response = requests.post(
        "http://localhost:8081/models", params=params, headers=header
    )
    file_content = Path(f"{REPO_ROOT}/key_file.json").read_text()
    print(file_content)

    yield "test"

    test_utils.stop_torchserve()


# Test describe model API with token enabled
def test_managament_api_with_token(setup_torchserve):
    key = read_key_file("management")
    header = {"Authorization": f"Bearer {key}"}
    response = requests.get("http://localhost:8081/models/mnist", headers=header)

    assert response.status_code == 200, "Token check failed"


# Test describe model API with incorrect token and no token
def test_managament_api_with_incorrect_token(setup_torchserve):
    # Using random key
    header = {"Authorization": "Bearer abcd1234"}
    response = requests.get(f"http://localhost:8081/models/mnist", headers=header)

    assert response.status_code == 400, "Token check failed"


# Test inference API with token enabled
def test_inference_api_with_token(setup_torchserve):
    key = read_key_file("inference")
    header = {"Authorization": f"Bearer {key}"}

    response = requests.post(
        url="http://localhost:8080/predictions/mnist",
        files={"data": open(data_file_zero, "rb")},
        headers=header,
    )

    assert response.status_code == 200, "Token check failed"


# Test inference API with incorrect token
def test_inference_api_with_incorrect_token(setup_torchserve):
    # Using random key
    header = {"Authorization": "Bearer abcd1234"}

    response = requests.post(
        url="http://localhost:8080/predictions/mnist",
        files={"data": open(data_file_zero, "rb")},
        headers=header,
    )

    assert response.status_code == 400, "Token check failed"


# Test Token API for regenerating new inference key
def test_token_inference_api(setup_torchserve):
    token_key = read_key_file("token")
    inference_key = read_key_file("inference")
    header_inference = {"Authorization": f"Bearer {inference_key}"}
    header_token = {"Authorization": f"Bearer {token_key}"}
    params = {"type": "inference"}

    # check inference works with current token
    response = requests.post(
        url="http://localhost:8080/predictions/mnist",
        files={"data": open(data_file_zero, "rb")},
        headers=header_inference,
    )
    assert response.status_code == 200, "Token check failed"

    # generate new inference token and check it is different
    response = requests.get(
        url="http://localhost:8081/token", params=params, headers=header_token
    )
    assert response.status_code == 200, "Token check failed"
    assert inference_key != read_key_file("inference"), "Key file not updated"

    # check inference does not works with original token
    response = requests.post(
        url="http://localhost:8080/predictions/mnist",
        files={"data": open(data_file_zero, "rb")},
        headers=header_inference,
    )
    assert response.status_code == 400, "Token check failed"


# Test Token API for regenerating new management key
def test_token_management_api(setup_torchserve):
    token_key = read_key_file("token")
    management_key = read_key_file("management")
    header = {"Authorization": f"Bearer {token_key}"}
    params = {"type": "management"}

    response = requests.get(
        url="http://localhost:8081/token", params=params, headers=header
    )

    assert management_key != read_key_file("management"), "Key file not updated"
    assert response.status_code == 200, "Token check failed"


# Test expiration time
@pytest.mark.module2
def test_token_expiration_time(setup_torchserve_expiration):
    key = read_key_file("management")
    header = {"Authorization": f"Bearer {key}"}
    response = requests.get("http://localhost:8081/models/mnist", headers=header)
    assert response.status_code == 200, "Token check failed"

    time.sleep(15)

    response = requests.get("http://localhost:8081/models/mnist", headers=header)
    assert response.status_code == 400, "Token check failed"
