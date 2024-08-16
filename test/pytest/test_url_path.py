import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import pytest
import requests
import test_utils
from model_archiver import ModelArchiverConfig

CURR_FILE_PATH = Path(__file__).parent
REPO_ROOT_DIR = CURR_FILE_PATH.parents[1]

HANDLER_PY = """
import logging
import torch
from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__file__)

class customHandler(BaseHandler):

    def initialize(self, context):
        self.context = context
        pass

    def preprocess(self, data):
        reqs = []
        for i in range(len(data)):
            reqs.append(self.context.get_request_header(i, "url_path"))
        return reqs

    def inference(self, data):
        return data

    def postprocess(self, data):
        return data
"""

MODEL_CONFIG_YAML = """
    #frontend settings
    # TorchServe frontend parameters
    minWorkers: 1
    batchSize: 2
    maxBatchDelay: 2000
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

    handler_py_file = work_dir / "handler.py"
    handler_py_file.write_text(HANDLER_PY)

    config = ModelArchiverConfig(
        model_name=model_name,
        version="1.0",
        serialized_file=None,
        model_file=None,
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
def register_model(mar_file_path, model_store, torchserve):
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
        ("batch_size", "2"),
        ("max_batch_delay", "2000"),
    )

    test_utils.reg_resp = test_utils.register_model_with_params(params)

    yield model_name

    test_utils.unregister_model(model_name)


def test_url_paths(model_name):
    response = requests.get(f"http://localhost:8081/models/{model_name}")
    assert response.status_code == 200, "Describe Failed"

    response = requests.get(
        f"http://localhost:8080/predictions/{model_name}/1.0/v1/chat/completion",
        json={"prompt": "Hello world"},
    )

    url_paths = ["v1/chat/completion", "v1/completion"]

    with ThreadPoolExecutor(max_workers=2) as e:
        futures = []
        for p in url_paths:

            def send_file(url):
                return requests.post(
                    f"http://localhost:8080/predictions/{model_name}/1.0/" + url,
                    json={"prompt": "Hello world"},
                )

            futures += [e.submit(send_file, p)]

    for i, f in enumerate(futures):
        prediction = f.result()
        assert prediction.content.decode("utf-8") == url_paths[i], "Wrong prediction"
