import json
import shutil
import time
from pathlib import Path
from unittest.mock import patch

import pytest
import requests
import test_utils
from model_archiver import ModelArchiverConfig

CURR_FILE_PATH = Path(__file__).parent
REPO_ROOT_DIR = CURR_FILE_PATH.parents[1]
config_file = REPO_ROOT_DIR / "test/resources/config_token.properties"
data_file_zero = REPO_ROOT_DIR / "test/pytest/test_data/0.png"
handler_py_file = REPO_ROOT_DIR / "test/pytest/test_data/session_handler.py"
model_py_file = REPO_ROOT_DIR / "examples/image_classifier/mnist/mnist.py"
model_pt_file = REPO_ROOT_DIR / "examples/image_classifier/mnist/mnist_cnn.pt"
metrics_yaml_file = REPO_ROOT_DIR / "ts/configs/metrics.yaml"
session_py_file = REPO_ROOT_DIR / "ts/metrics/sessions.py"


HANDLER_PY = """
import torch
from ts.torch_handler.base_handler import BaseHandler

class customHandler(BaseHandler):

    def initialize(self, context):
        super().initialize(context)
"""

MODEL_CONFIG_YAML = """
    #frontend settings
    # TorchServe frontend parameters
    minWorkers: 1
    batchSize: 1
    maxWorkers: 1
    """


@pytest.fixture(scope="module")
def model_name():
    yield "some_model"


@pytest.fixture(scope="module")
def work_dir(tmp_path_factory, model_name):
    return Path(tmp_path_factory.mktemp(model_name))


@pytest.fixture(scope="module", name="mar_file_path")
def create_mar_file(work_dir, model_archiver, model_name):
    mar_file_path = work_dir.joinpath(model_name + ".mar")

    model_config_yaml_file = work_dir / "model_config.yaml"
    model_config_yaml_file.write_text(MODEL_CONFIG_YAML)

    config = ModelArchiverConfig(
        model_name=model_name,
        version="1.0",
        serialized_file=model_pt_file.as_posix(),
        model_file=model_py_file.as_posix(),
        handler=handler_py_file.as_posix(),
        extra_files=None,
        export_path=work_dir,
        requirements_file=None,
        runtime="python",
        force=False,
        archive_format="default",
        config_file=model_config_yaml_file.as_posix(),
    )

    with patch("archiver.ArgParser.export_model_args_parser", return_value=config):
        model_archiver.generate_model_archive()

        assert mar_file_path.exists()

        yield mar_file_path.as_posix()

        # Clean up files

    mar_file_path.unlink(missing_ok=True)

    # Clean up files


@pytest.fixture(scope="module", name="model_name")
def register_model(mar_file_path, model_store, work_dir, model_name, torchserve):
    """
    Register the model in torchserve
    """
    shutil.copy(mar_file_path, model_store)

    file_name = Path(mar_file_path).name

    model_name = Path(file_name).stem

    params = (
        ("model_name", model_name),
        ("url", file_name),
        ("initial_workers", "1"),
        ("synchronous", "true"),
        ("batch_size", "1"),
    )

    CONFIG_PROPERTIES = f"""
    system_metrics_cmd={session_py_file.as_posix()} --model_name {model_name} --timeout 2
    metrics_config={metrics_yaml_file.as_posix()}
    metric_time_interval=1
    """

    config_properties_file = work_dir / "config.properties"
    config_properties_file.write_text(CONFIG_PROPERTIES)

    stdout = test_utils.start_torchserve(
        model_store=model_store,
        snapshot_file=config_properties_file.as_posix(),
        gen_mar=False,
    )

    test_utils.reg_resp = test_utils.register_model_with_params(params)

    yield model_name, stdout

    test_utils.unregister_model(model_name)

    test_utils.stop_torchserve()


def test_open_close(model_name):
    model_name, stdout = model_name
    # Open two sessions
    # Opening the second session will close the first (checked through OpenSession metric)
    # Waiting timeout will close session two as well

    response = requests.get(f"http://localhost:8081/models/{model_name}")
    assert response.status_code == 200, "Describe Failed"

    # Open session 1
    response_open_1 = requests.post(
        f"http://localhost:8080/predictions/{model_name}",
        data=json.dumps({"request_type": "open_session"}),
    )
    assert response_open_1.status_code == 200, "Open 1 Failed"

    res_1 = json.loads(response_open_1.content)
    assert "session_id" in res_1 and "msg" in res_1

    # Open session 2
    response_open_2 = requests.post(
        f"http://localhost:8080/predictions/{model_name}",
        data=json.dumps({"request_type": "open_session"}),
    )
    assert response_open_2.status_code == 200, "Open 2 Failed"

    res_2 = json.loads(response_open_2.content)
    assert "session_id" in res_2 and "msg" in res_2

    time.sleep(1)
    open_session_metric_output = []
    # Empty queue
    while not stdout.empty():
        out = stdout.get_nowait()
        if "OpenSession" in out:
            open_session_metric_output += [out]

    assert len(open_session_metric_output), "No metric output"
    assert (
        "0.0" in open_session_metric_output[0]
    ), "Open session was not 0 at the beginning"
    assert "1.0" in open_session_metric_output[-1], "Currently open sessions is not 1"

    time.sleep(2)
    while not stdout.empty():
        out = stdout.get_nowait()
        if "OpenSession" in out:
            open_session_metric_output += [out]
    assert (
        "0.0" in open_session_metric_output[-1]
    ), "Last session should have been timed out"
