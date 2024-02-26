import os
import shutil
import signal
from pathlib import Path
from subprocess import PIPE, STDOUT, Popen

import pytest
import requests

CURR_FILE_PATH = Path(__file__).parent

# Matching sample KServe config.properties sample
BASE_PORT = 8000
INFERENCE_PORT = BASE_PORT + 85
MANAGEMENT_PORT = BASE_PORT + 85
METRICS_PORT = BASE_PORT + 82

TS_CONFIG_CONTENT = f"""
inference_address=http://127.0.0.1:{INFERENCE_PORT}
management_address=http://127.0.0.1:{MANAGEMENT_PORT}
metrics_address=http://127.0.0.1:{METRICS_PORT}
install_py_dep_per_model=false
number_of_netty_threads=1
job_queue_size=1
service_envelope=kservev2
"""

KSERVE_CONFIG_CONTENT = f"""
inference_address=http://127.0.0.1:{INFERENCE_PORT}
management_address=http://127.0.0.1:{MANAGEMENT_PORT}
metrics_address=http://127.0.0.1:{METRICS_PORT}
grpc_inference_port=7070
grpc_management_port=7071
model_store=/
model_snapshot={{"name": "startup.cfg","modelCount": 1,"models": {{"test": {{"1.0": {{"defaultVersion": true, "marName": "test.mar", "minWorkers": 1, "maxWorkers": 5, "batchSize": 5, "maxBatchDelay": 200, "responseTimeout": 60}}}}}}}}
"""

HANDLER_PY = """
from abc import ABC
from ts.torch_handler.base_handler import BaseHandler

class DummyHandler(BaseHandler, ABC):

    def __init__(self):
        super().__init__()

    def initialize(self, context):
        self.initialized = True

    def preprocess(self, requests):
        return []

    def inference(self, samples, *args, **kwargs):
        return [{"test": 42}]

    def postprocess(self, data):
        return [data]
"""

MODELS_DIR = CURR_FILE_PATH / "models_gen"
MODEL_STORE = MODELS_DIR / "model_store"
CONFIG_DIR = MODELS_DIR / "config"

import subprocess


def setup_module(module):
    MODEL_STORE.mkdir(parents=True, exist_ok=True)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    handler_file = str(MODELS_DIR / "handler.py")
    with open(handler_file, "wt") as handler:
        handler.write(HANDLER_PY)

    ts_config = str(CONFIG_DIR / "ts_config.properties")
    with open(ts_config, "wt") as config:
        config.write(TS_CONFIG_CONTENT)

    subprocess.run(
        f"torch-model-archiver -f --model-name test --version 1.0 --export-path {MODEL_STORE} --handler {handler_file}",
        shell=True,
        check=True,
    )

    cmd = [
        "torchserve",
        "--start",
        "--ncs",
        "--model-store",
        MODEL_STORE,
        "--models",
        "test.mar",
        f"--ts-config",
        ts_config,
    ]
    p = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=STDOUT)
    for line in p.stdout:
        print(line.decode("utf8").strip())
        if "Model server started" in str(line).strip():
            break


def teardown_module(module):
    subprocess.run("torchserve --stop", shell=True, check=True)
    shutil.rmtree(MODELS_DIR)


@pytest.fixture(name="protobuf")
def generate_protobuf_code():
    root_path = CURR_FILE_PATH.parent.parent
    kserve_wrapper_path = root_path / "kubernetes/kserve/kserve_wrapper"
    frontend_path = root_path / "frontend/server/src/main/resources/proto"
    third_party = root_path / "third_party/google/rpc/"

    cmd = [
        "python",
        "-m",
        "grpc_tools.protoc",
        f"--proto_path={frontend_path}",
        f"--proto_path={third_party}",
        f"--python_out={kserve_wrapper_path}",
        f"--grpc_python_out={kserve_wrapper_path}",
    ]

    for item in ["inference", "management"]:
        cmd.append(f"{frontend_path}/{item}.proto")

    subprocess.run(" ".join(cmd), shell=True, check=True)


@pytest.fixture
def kserve_wrapper(protobuf):
    kserve_config = str(CONFIG_DIR / "kserve_config.properties")
    with open(kserve_config, "wt") as config:
        config.write(KSERVE_CONFIG_CONTENT)

    cmd = ["python3", "kubernetes/kserve/kserve_wrapper/__main__.py"]
    env = os.environ.copy()
    env["CONFIG_PATH"] = kserve_config

    p = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=STDOUT, env=env)
    for line in p.stdout:
        print(line.decode("utf8").strip())
        cleaned_line = str(line).strip()
        if "Started server process" in cleaned_line:
            wrapper_pid = int(
                cleaned_line[cleaned_line.rfind("[") + 1 : cleaned_line.rfind("]")]
            )
        if "Application startup complete" in cleaned_line:
            break
    yield

    os.kill(wrapper_pid, signal.SIGINT)


def test_inference(kserve_wrapper):
    json_data = {
        "inputs": [{"name": "uuid", "shape": -1, "datatype": "BYTES", "data": ["test"]}]
    }

    response = requests.post(
        f"http://127.0.0.1:8080/v1/models/test:predict", json=json_data, timeout=120
    )

    assert response.status_code == 200
