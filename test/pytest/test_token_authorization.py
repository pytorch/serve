import json
import os
import shutil
import subprocess
import tempfile
import time

import requests
import test_utils

ROOT_DIR = os.path.join(tempfile.gettempdir(), "workspace")
REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../")
data_file_kitten = os.path.join(REPO_ROOT, "test/pytest/test_data/kitten.jpg")


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
    result = subprocess.run(
        f"python ts_scripts/install_from_source",
        shell=True,
        capture_output=True,
        text=True,
    )


# Parse json file and return key
def read_key_file(type):
    json_file_path = os.path.join(REPO_ROOT, "key_file.json")
    with open(json_file_path) as json_file:
        json_data = json.load(json_file)

    # Extract the three keys
    management_key = None
    inference_key = None
    api_key = None
    for key_string in json_data:
        if "Management Key" in key_string:
            management_key = key_string.split(":")[1].strip().split("---")[0].strip()
        elif "Inference Key" in key_string:
            inference_key = key_string.split(":")[1].strip().split("---")[0].strip()
        elif "API Key" in key_string:
            api_key = key_string.split(":")[1].strip().split("---")[0].strip()

    options = {
        "management": management_key,
        "inference": inference_key,
        "token": api_key,
    }
    key = options.get(type, "Invalid data type")
    return key


def setup_torchserve():
    get_plugin_jar()
    MODEL_STORE = os.path.join(ROOT_DIR, "model_store/")
    PLUGIN_STORE = os.path.join(ROOT_DIR, "plugins-path")

    test_utils.start_torchserve(no_config_snapshots=True, plugin_folder=PLUGIN_STORE)
    time.sleep(10)

    key = read_key_file("management")
    header = {"Authorization": f"Bearer {key}"}

    params = (
        ("model_name", "resnet18"),
        ("url", "resnet-18.mar"),
        ("initial_workers", "1"),
        ("synchronous", "true"),
    )
    response = requests.post(
        "http://localhost:8081/models", params=params, headers=header
    )
    time.sleep(5)
    print("register reponse")
    print(response.text)

    result = subprocess.run(
        f"cat {REPO_ROOT}/key_file.json",
        shell=True,
        capture_output=True,
        text=True,
    )
    print("Curl output:")
    print(result.stdout)


# Test describe model API with token enabled
def test_managament_api_with_token():
    test_utils.stop_torchserve()
    setup_torchserve()
    key = read_key_file("management")
    header = {"Authorization": f"Bearer {key}"}
    print(key)
    response = requests.get("http://localhost:8081/models/resnet18", headers=header)
    time.sleep(5)
    print(response.text)

    assert response.status_code == 200, "Token check failed"


# Test describe model API with incorrect token and no token
def test_managament_api_with_incorrect_token():
    # Using random key
    header = {"Authorization": "Bearer abcd1234"}

    response = requests.get(f"http://localhost:8081/models/resnet18", headers=header)
    time.sleep(5)
    print(response.text)

    assert response.status_code == 400, "Token check failed"


# Test inference API with token enabled
def test_inference_api_with_token():
    key = read_key_file("inference")
    header = {"Authorization": f"Bearer {key}"}

    response = requests.post(
        url="http://localhost:8080/predictions/resnet18",
        files={"data": open(data_file_kitten, "rb")},
        headers=header,
    )
    time.sleep(5)
    print(response.text)
    print(key)

    assert response.status_code == 200, "Token check failed"


# Test inference API with incorrect token
def test_inference_api_with_incorrect_token():
    # Using random key
    header = {"Authorization": "Bearer abcd1234"}

    response = requests.post(
        url="http://localhost:8080/predictions/resnet18",
        files={"data": open(data_file_kitten, "rb")},
        headers=header,
    )
    time.sleep(5)
    print(response.text)

    assert response.status_code == 400, "Token check failed"


# Test Token API for regenerating new inference key
def test_token_inference_api():
    token_key = read_key_file("token")
    inference_key = read_key_file("inference")
    header_inference = {"Authorization": f"Bearer {inference_key}"}
    header_token = {"Authorization": f"Bearer {token_key}"}
    params = {"type": "inference"}

    # check inference works with current token
    response = requests.post(
        url="http://localhost:8080/predictions/resnet18",
        files={"data": open(data_file_kitten, "rb")},
        headers=header_inference,
    )
    time.sleep(5)
    assert response.status_code == 200, "Token check failed"

    # generate new inference token and check it is different
    response = requests.get(
        url="http://localhost:8081/token", params=params, headers=header_token
    )
    time.sleep(5)
    print(response.text)
    print(token_key)
    assert response.status_code == 200, "Token check failed"
    assert inference_key != read_key_file("inference"), "Key file not updated"

    # check inference does not works with original token
    response = requests.post(
        url="http://localhost:8080/predictions/resnet18",
        files={"data": open(data_file_kitten, "rb")},
        headers=header_inference,
    )
    time.sleep(5)
    assert response.status_code == 400, "Token check failed"


# Test Token API for regenerating new management key
def test_token_management_api():
    token_key = read_key_file("token")
    management_key = read_key_file("management")
    header = {"Authorization": f"Bearer {token_key}"}
    params = {"type": "management"}

    response = requests.get(
        url="http://localhost:8081/token", params=params, headers=header
    )
    time.sleep(5)
    print(response.text)
    print(token_key)

    assert management_key != read_key_file("management"), "Key file not updated"
    assert response.status_code == 200, "Token check failed"
    test_utils.stop_torchserve()
