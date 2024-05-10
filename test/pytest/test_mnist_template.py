import shutil
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
handler_py_file = REPO_ROOT_DIR / "examples/image_classifier/mnist/mnist_handler.py"
model_py_file = REPO_ROOT_DIR / "examples/image_classifier/mnist/mnist.py"
model_pt_file = REPO_ROOT_DIR / "examples/image_classifier/mnist/mnist_cnn.pt"


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
        ("batch_size", "1"),
    )

    test_utils.reg_resp = test_utils.register_model_with_params(params)

    yield model_name

    test_utils.unregister_model(model_name)


def test_mnist_template(model_name):
    response = requests.get(f"http://localhost:8081/models/{model_name}")
    assert response.status_code == 200, "Describe Failed"

    with open(data_file_zero, "rb") as f:
        response = requests.post(
            f"http://localhost:8080/predictions/{model_name}",
            data=f,
        )

    assert response.content.decode("utf-8") == "0", "Wrong prediction"
