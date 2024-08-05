import shutil
import time
from pathlib import Path
from unittest.mock import patch

import pytest
import requests
import test_utils
from model_archiver import ModelArchiverConfig

REPO_ROOT_DIR = Path(__file__).parents[2]
EXAMPLES_DIR = REPO_ROOT_DIR / "examples"
MNIST_DIR = EXAMPLES_DIR / "image_classifier" / "mnist"

model_pt_file = MNIST_DIR / "mnist_cnn.pt"
model_py_file = MNIST_DIR / "mnist.py"

HANDLER_PY = """
import time
from ts.torch_handler.base_handler import BaseHandler

class CustomHandler(BaseHandler):
    def initialize(self, context):
        time.sleep(10) # to simulate long startup time
        super().initialize(context)

    def handle(self, data, context):
        return ["Dummy response"]
"""

MODEL_CONFIG_YAML_TEMPLATE = """
minWorkers: 1
maxWorkers: 1
responseTimeout: 1
startupTimeout: {startup_timeout}
"""


@pytest.fixture(scope="module")
def model_name():
    return "startup_timeout_test_model"


@pytest.fixture(scope="module")
def work_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("startup_timeout_test")


@pytest.fixture(scope="module", name="mar_file_path", params=[30, 5])
def create_mar_file(work_dir, model_archiver, model_name, request):
    startup_timeout = request.param
    model_name = model_name + f"_{startup_timeout}"
    mar_file_path = work_dir / f"{model_name}.mar"

    handler_file = work_dir / f"handler_{startup_timeout}.py"
    handler_file.write_text(HANDLER_PY)

    model_config_file = work_dir / f"model_config_{startup_timeout}.yaml"
    model_config_yaml = MODEL_CONFIG_YAML_TEMPLATE.format(
        startup_timeout=startup_timeout
    )
    model_config_file.write_text(model_config_yaml)

    config = ModelArchiverConfig(
        model_name=model_name,
        version="1.0",
        serialized_file=str(model_pt_file),
        model_file=str(model_py_file),
        handler=str(handler_file),
        export_path=str(work_dir),
        config_file=str(model_config_file),
    )

    with patch("archiver.ArgParser.export_model_args_parser", return_value=config):
        model_archiver.generate_model_archive()

    assert mar_file_path.exists()
    return str(mar_file_path)


@pytest.fixture(scope="function")
def register_model(mar_file_path, model_store, torchserve, request):
    shutil.copy(mar_file_path, model_store)
    model_name = Path(mar_file_path).stem
    startup_timeout = request.node.callspec.params["mar_file_path"]

    params = (
        ("model_name", model_name),
        ("url", Path(mar_file_path).name),
        ("initial_workers", "1"),
        ("synchronous", "false"),
    )

    response = test_utils.register_model_with_params(params)
    assert response.status_code == 202, "Model registration failed"

    yield model_name, torchserve, startup_timeout

    test_utils.unregister_model(model_name)


def test_startup_timeout(register_model):
    model_name, torchserve, startup_timeout = register_model

    model_status = "LOADING"
    max_wait = 30
    start_time = time.time()

    # We expect model to timeout in this case, since in handler initialization we set time.sleep(10)
    if startup_timeout == 5:
        expected_error = "org.pytorch.serve.wlm.WorkerInitializationException: Backend worker did not respond in given time"
        while (time.time() - start_time) < max_wait:
            output = torchserve.get()
            if "Backend worker error" in output:
                error_message = torchserve.get()
                break
        else:
            assert (
                False
            ), f"Timeout waiting for 'Backend worker error' (waited {max_wait} seconds)"
        assert (
            expected_error in error_message
        ), "Unexpected error message, expected model to timeout during startup"
        return
    # If startup timeout is set to 30 then we expect model to startup, even though
    # response timeout is set to 1
    while (time.time() - start_time) < max_wait:
        response = requests.get(f"http://localhost:8081/models/{model_name}")
        if response.status_code == 200:
            model_status = response.json()[0]["workers"][0]["status"]
            if model_status == "READY":
                break
        time.sleep(1)

    end_time = time.time()
    elapsed_time = end_time - start_time

    assert response.status_code == 200, "Model startup failed"
    assert model_status == "READY", f"Unexpected model status: {model_status}"
    assert 10 <= elapsed_time < 20, f"Unexpected startup time: {elapsed_time} seconds"
