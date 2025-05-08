import glob
import importlib.util
import json
import os
import subprocess
import time
from pathlib import Path

import pytest

try:
    import rebel  # nopycln: import

    RBLN_AVAILABLE = True
except ImportError:
    RBLN_AVAILABLE = False

REQUIRED_PKG = ["torchvision", "torch"]

CURR_FILE_PATH = Path(__file__).parent
RBLN_TEST_DATA_DIR = os.path.join(CURR_FILE_PATH, "test_data", "rbln_compile")

COMPILE_FILE = os.path.join(RBLN_TEST_DATA_DIR, "compile.py")
HANDLER_FILE = os.path.join(RBLN_TEST_DATA_DIR, "rbln_handler.py")
CONFIG_PROPERTIES = os.path.join(RBLN_TEST_DATA_DIR, "config.properties")
SERIALIZED_FILE = os.path.join(RBLN_TEST_DATA_DIR, "resnet50.rbln")
MODEL_STORE_DIR = os.path.join(RBLN_TEST_DATA_DIR, "model_store")
MODEL_NAME = "resnet50"


@pytest.fixture(scope="session", autouse=True)
def install_pkgs():
    for package_name in REQUIRED_PKG:
        if importlib.util.find_spec(package_name) is None:
            print(f"Installing missing package: {package_name}")
            try:
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", package_name]
                )
            except subprocess.CalledProcessError as e:
                pytest.fail(f"Fail to install package {package_name}")
            print(f"Installing missing package: {package_name} - Installed.")


def ensure_package(import_name, package_name):
    if importlib.util.find_spec(import_name) is None:
        print(f"Fail to find RBLN package : {package_name}")
        return False
    return True


@pytest.mark.skipif(RBLN_AVAILABLE == False, reason='"rebel-compiler" is not installed')
class TestTorchRbln:
    def teardown_class(self):
        subprocess.run("torchserve --stop", shell=True, check=True)
        time.sleep(10)

    def test_rbln_sdk_packages(self):
        assert ensure_package("rebel", "rebel-compiler") == True

    def test_archive_model_artifact(self):
        assert len(glob.glob(COMPILE_FILE)) == 1
        assert len(glob.glob(HANDLER_FILE)) == 1
        assert len(glob.glob(CONFIG_PROPERTIES)) == 1

        subprocess.run(
            f"cd {RBLN_TEST_DATA_DIR} && python3 {COMPILE_FILE}", shell=True, check=True
        )
        subprocess.run(f"mkdir -p {MODEL_STORE_DIR}", shell=True, check=True)

        assert len(glob.glob(SERIALIZED_FILE)) == 1

        subprocess.run(
            f"torch-model-archiver --model-name {MODEL_NAME} --version 1.0 --handler {HANDLER_FILE} --serialized-file {SERIALIZED_FILE} --export-path {MODEL_STORE_DIR} -f",
            shell=True,
            check=True,
        )
        assert len(glob.glob(os.path.join(MODEL_STORE_DIR, f"{MODEL_NAME}.mar"))) == 1

    def test_start_torchserve(self):
        subprocess.run(
            f"torchserve --start --ncs --models {MODEL_NAME}.mar --model-store {MODEL_STORE_DIR} --ts-config {CONFIG_PROPERTIES} --disable-token-auth",
            shell=True,
            check=True,
        )
        time.sleep(10)
        assert len(glob.glob("logs/access_log.log")) == 1
        assert len(glob.glob("logs/model_log.log")) == 1
        assert len(glob.glob("logs/ts_log.log")) == 1

    def test_server_status(self):
        result = subprocess.run(
            "curl http://localhost:8080/ping",
            shell=True,
            capture_output=True,
            check=True,
        )
        expected_server_status_str = '{"status": "Healthy"}'
        expected_server_status = json.loads(expected_server_status_str)
        assert json.loads(result.stdout) == expected_server_status

    def test_registered_model(self):
        result = subprocess.run(
            "curl http://localhost:8081/models",
            shell=True,
            capture_output=True,
            check=True,
        )
        expected_registered_model_str = (
            '{"models": [{"modelName": "resnet50", "modelUrl": "resnet50.mar"}]}'
        )
        expected_registered_model = json.loads(expected_registered_model_str)
        assert json.loads(result.stdout) == expected_registered_model

    def test_serve_inference(self):
        if not Path(f"{RBLN_TEST_DATA_DIR}/tabby.jpg").exists():
            subprocess.run(
                f"cd {RBLN_TEST_DATA_DIR} && wget https://rbln-public.s3.ap-northeast-2.amazonaws.com/images/tabby.jpg",
                shell=True,
            )
        result = subprocess.run(
            f'cd {RBLN_TEST_DATA_DIR} && curl -X POST "http://127.0.0.1:8080/predictions/resnet50" -H "Content-Type: application/octet-stream" --data-binary @./tabby.jpg',
            shell=True,
            capture_output=True,
            check=True,
        )
        expected_result_str = '{"result":"tabby"}'
        expected_result = json.loads(expected_result_str)
        assert json.loads(result.stdout) == expected_result
