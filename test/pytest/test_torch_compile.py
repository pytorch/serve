import glob
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import pytest
import torch
from pkg_resources import packaging

PT_2_AVAILABLE = (
    True
    if packaging.version.parse(torch.__version__) >= packaging.version.parse("2.0")
    else False
)


CURR_FILE_PATH = Path(__file__).parent
TEST_DATA_DIR = os.path.join(CURR_FILE_PATH, "test_data", "torch_compile")

MODEL_FILE = os.path.join(TEST_DATA_DIR, "model.py")
HANDLER_FILE = os.path.join(TEST_DATA_DIR, "compile_handler.py")
YAML_CONFIG = os.path.join(TEST_DATA_DIR, "pt2.yaml")


SERIALIZED_FILE = os.path.join(TEST_DATA_DIR, "model.pt")
MODEL_STORE_DIR = os.path.join(TEST_DATA_DIR, "model_store")
MODEL_NAME = "half_plus_two"


@pytest.mark.skipif(
    platform.system() != "Linux", reason="Skipping test on non-Linux system"
)
@pytest.mark.skipif(PT_2_AVAILABLE == False, reason="torch version is < 2.0.0")
class TestTorchCompile:
    def teardown_class(self):
        subprocess.run("torchserve --stop", shell=True, check=True)
        time.sleep(10)

    def test_archive_model_artifacts(self):
        assert len(glob.glob(MODEL_FILE)) == 1
        assert len(glob.glob(YAML_CONFIG)) == 1
        subprocess.run(f"cd {TEST_DATA_DIR} && python model.py", shell=True, check=True)
        subprocess.run(f"mkdir -p {MODEL_STORE_DIR}", shell=True, check=True)
        subprocess.run(
            f"torch-model-archiver --model-name {MODEL_NAME} --version 1.0 --model-file {MODEL_FILE} --serialized-file {SERIALIZED_FILE} --config-file {YAML_CONFIG} --export-path {MODEL_STORE_DIR} --handler {HANDLER_FILE} -f",
            shell=True,
            check=True,
        )
        assert len(glob.glob(SERIALIZED_FILE)) == 1
        assert len(glob.glob(os.path.join(MODEL_STORE_DIR, f"{MODEL_NAME}.mar"))) == 1

    def test_start_torchserve(self):
        cmd = f"torchserve --start --ncs --models {MODEL_NAME}.mar --model-store {MODEL_STORE_DIR}"
        subprocess.run(
            cmd,
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
        expected_registered_model_str = '{"models": [{"modelName": "half_plus_two", "modelUrl": "half_plus_two.mar"}]}'
        expected_registered_model = json.loads(expected_registered_model_str)
        assert json.loads(result.stdout) == expected_registered_model

    def test_serve_inference(self):
        request_data = {"instances": [[1.0], [2.0], [3.0]]}
        request_json = json.dumps(request_data)

        result = subprocess.run(
            f"curl -s -X POST -H \"Content-Type: application/json;\" http://localhost:8080/predictions/half_plus_two -d '{request_json}'",
            shell=True,
            capture_output=True,
            check=True,
        )

        string_result = result.stdout.decode("utf-8")
        float_result = float(string_result)
        expected_result = 3.5

        assert float_result == expected_result

        model_log_path = glob.glob("logs/model_log.log")[0]
        with open(model_log_path, "rt") as model_log_file:
            model_log = model_log_file.read()
            assert "Compiled model with backend inductor" in model_log
