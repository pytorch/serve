import glob
import json
import os
import subprocess
import time
from pathlib import Path

import packaging.version
import pytest

try:
    import torch_xla

    TORCHXLA_AVAILABLE = (
        True
        if packaging.version.parse(torch_xla.__version__)
        >= packaging.version.parse("2.0")
        else False
    )
except:
    TORCHXLA_AVAILABLE = False

CURR_FILE_PATH = Path(__file__).parent
TORCH_XLA_TEST_DATA_DIR = os.path.join(CURR_FILE_PATH, "test_data", "torch_compile")

MODEL_FILE = os.path.join(TORCH_XLA_TEST_DATA_DIR, "model.py")
YAML_CONFIG = os.path.join(TORCH_XLA_TEST_DATA_DIR, "xla.yaml")
CONFIG_PROPERTIES = os.path.join(TORCH_XLA_TEST_DATA_DIR, "config.properties")

SERIALIZED_FILE = os.path.join(TORCH_XLA_TEST_DATA_DIR, "model.pt")
MODEL_STORE_DIR = os.path.join(TORCH_XLA_TEST_DATA_DIR, "model_store")
MODEL_NAME = "half_plus_two"


@pytest.mark.skipif(TORCHXLA_AVAILABLE == False, reason="PyTorch/XLA is not installed")
class TestTorchXLA:
    def teardown_class(self):
        subprocess.run("torchserve --stop", shell=True, check=True)
        time.sleep(10)

    def test_archive_model_artifacts(self):
        assert len(glob.glob(MODEL_FILE)) == 1
        assert len(glob.glob(YAML_CONFIG)) == 1
        assert len(glob.glob(CONFIG_PROPERTIES)) == 1
        subprocess.run(
            f"cd {TORCH_XLA_TEST_DATA_DIR} && python model.py", shell=True, check=True
        )
        subprocess.run(f"mkdir -p {MODEL_STORE_DIR}", shell=True, check=True)
        subprocess.run(
            f"torch-model-archiver --model-name {MODEL_NAME} --version 1.0 --model-file {MODEL_FILE} --serialized-file {SERIALIZED_FILE} --config-file {YAML_CONFIG} --export-path {MODEL_STORE_DIR} --handler base_handler -f",
            shell=True,
            check=True,
        )
        assert len(glob.glob(SERIALIZED_FILE)) == 1
        assert len(glob.glob(os.path.join(MODEL_STORE_DIR, f"{MODEL_NAME}.mar"))) == 1

    def test_start_torchserve(self):
        subprocess.run(
            f"torchserve --start --ncs --models {MODEL_NAME}.mar --model-store {MODEL_STORE_DIR} --ts-config {CONFIG_PROPERTIES}",
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
        request = "'{\"" 'instances"' ": [[1.0], [2.0], [3.0]]}'"
        result = subprocess.run(
            f'curl -s -X POST -H "Content-Type: application/json;" http://localhost:8080/predictions/half_plus_two -d {request}',
            shell=True,
            capture_output=True,
            check=True,
        )
        expected_result_str = '{"predictions": [[2.5], [3.0], [3.5]]}'
        expected_result = json.loads(expected_result_str)
        assert json.loads(result.stdout) == expected_result

        model_log_path = glob.glob("logs/model_log.log")[0]
        with open(model_log_path, "rt") as model_log_file:
            model_log = model_log_file.read()
            assert "Compiled model with backend torchxla_trace_once" in model_log
            assert "done compiler function torchxla_trace_once" in model_log
