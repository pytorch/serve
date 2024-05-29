import json
import shutil
import time
from pathlib import Path
from string import Template
from unittest.mock import patch

import pytest
import requests
import test_utils
import yaml
from model_archiver import ModelArchiverConfig

CURR_FILE_PATH = Path(__file__).parent
REPO_ROOT_DIR = CURR_FILE_PATH.parents[1]
config_file = REPO_ROOT_DIR / "test/resources/config_token.properties"
data_file_zero = REPO_ROOT_DIR / "test/pytest/test_data/0.png"
handler_py_file = REPO_ROOT_DIR / "test/pytest/test_data/session_handler.py"
model_py_file = REPO_ROOT_DIR / "examples/image_classifier/mnist/mnist.py"
model_pt_file = REPO_ROOT_DIR / "examples/image_classifier/mnist/mnist_cnn.pt"
metrics_yaml_file = REPO_ROOT_DIR / "ts/configs/metrics.yaml"


HANDLER_PY = """
import torch
from ts.torch_handler.base_handler import BaseHandler

class customHandler(BaseHandler):

    def initialize(self, context):
        super().initialize(context)
"""


METRIC_COLLECTOR = Template(
    """
import logging
import sys
from pathlib import Path
from torch.distributed import FileStore

from ts.metrics.dimension import Dimension
from ts.metrics.metric import Metric

if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, format="%(message)s", level=logging.INFO)

    store_path = Path("/tmp") / "${MODEL_NAME}_store"
    store = FileStore(store_path.as_posix(), -1)

    open_sessions_num = len(store.get("open_sessions").decode("utf-8").split(";"))

    dimension = [Dimension("Level", "Host")]
    logging.info(str(Metric("OpenSessions", open_sessions_num, "count", dimension)))
    logging.info("")
"""
)

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

    COLLECTOR_PY = work_dir / "metric_collector.py"
    COLLECTOR_PY.write_text(METRIC_COLLECTOR.substitute({"MODEL_NAME": model_name}))

    with open(metrics_yaml_file) as f:
        metrics_config = yaml.safe_load(f)
    metrics_config["ts_metrics"]["counter"] += [
        {"name": "OpenSessions", "unit": "Count", "dimensions": ["hostname"]}
    ]

    METRICS_YAML = work_dir / "metrics.yaml"
    with open(METRICS_YAML, "w") as f:
        yaml.safe_dump(metrics_config, f)

    CONFIG_PROPERTIES = f"""
    system_metrics_cmd={COLLECTOR_PY.as_posix()}
    metrics_config={METRICS_YAML.as_posix()}
    """

    config_properties_file = work_dir / "config.properties"
    config_properties_file.write_text(CONFIG_PROPERTIES)

    test_utils.start_torchserve(
        model_store=model_store,
        snapshot_file=config_properties_file.as_posix(),
        gen_mar=False,
    )

    test_utils.reg_resp = test_utils.register_model_with_params(params)

    yield model_name

    test_utils.unregister_model(model_name)

    test_utils.stop_torchserve()


def test_open_close(model_name):
    # Open two sessions
    # Try closing session 2 twice
    # Wait for session 1 to timeout, then open sesison 3
    # Try closing session 1 but it should have been closed during opening of 3
    response = requests.get(f"http://localhost:8081/models/{model_name}")
    assert response.status_code == 200, "Describe Failed"

    response_open_1 = requests.post(
        f"http://localhost:8080/predictions/{model_name}",
        data=json.dumps({"request_type": "open_session"}),
    )
    assert response_open_1.status_code == 200, "Open 1 Failed"

    res_1 = json.loads(response_open_1.content)
    assert "session_id" in res_1 and "msg" in res_1

    response_open_2 = requests.post(
        f"http://localhost:8080/predictions/{model_name}",
        data=json.dumps({"request_type": "open_session"}),
    )
    assert response_open_2.status_code == 200, "Open 2 Failed"

    res_2 = json.loads(response_open_2.content)
    assert "session_id" in res_2 and "msg" in res_2

    response_close_2 = requests.post(
        f"http://localhost:8080/predictions/{model_name}",
        data=json.dumps(
            {"request_type": "close_session", "session_id": res_2["session_id"]}
        ),
    )
    assert response_close_2.status_code == 200, "Close 2 Failed"

    response_double_close_2 = requests.post(
        f"http://localhost:8080/predictions/{model_name}",
        data=json.dumps(
            {"request_type": "close_session", "session_id": res_2["session_id"]}
        ),
    )
    assert response_double_close_2.status_code == 200, "Double close 2 Failed"

    res_3 = json.loads(response_double_close_2.content)
    assert "session_id" in res_3 and "msg" in res_3
    assert res_3["msg"] == "Session was already closed"

    response_close_1 = requests.post(
        f"http://localhost:8080/predictions/{model_name}",
        data=json.dumps(
            {"request_type": "close_session", "session_id": res_1["session_id"]}
        ),
    )
    assert response_close_1.status_code == 200, "Close 2 Failed"

    time.sleep(3)

    response_open_3 = requests.post(
        f"http://localhost:8080/predictions/{model_name}",
        data=json.dumps({"request_type": "open_session"}),
    )
    assert response_open_3.status_code == 200, "Open 3 Failed"

    response_close__after_timeout_1 = requests.post(
        f"http://localhost:8080/predictions/{model_name}",
        data=json.dumps(
            {"request_type": "close_session", "session_id": res_1["session_id"]}
        ),
    )
    assert (
        response_close__after_timeout_1.status_code == 200
    ), "Close 1 after timeout Failed"
    res_4 = json.loads(response_close__after_timeout_1.content)
    assert "session_id" in res_4 and "msg" in res_4
    assert res_4["msg"] == "Session was already closed"
