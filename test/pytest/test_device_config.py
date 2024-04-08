import os
import platform
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import requests
import test_utils
from model_archiver import ModelArchiverConfig

CURR_FILE_PATH = Path(__file__).parent
REPO_ROOT_DIR = CURR_FILE_PATH.parent.parent
ROOT_DIR = os.path.join(tempfile.gettempdir(), "workspace")
REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../")
data_file_zero = os.path.join(REPO_ROOT, "test/pytest/test_data/0.png")
config_file = os.path.join(REPO_ROOT, "test/resources/config_token.properties")
mnist_scriptes_py = os.path.join(REPO_ROOT, "examples/image_classifier/mnist/mnist.py")

HANDLER_PY = """
from ts.torch_handler.base_handler import BaseHandler

class deviceHandler(BaseHandler):

    def initialize(self, context):
        super().initialize(context)
        assert self.get_device().type == "mps"
"""

MODEL_CONFIG_YAML = """
    #frontend settings
    # TorchServe frontend parameters
    minWorkers: 1
    batchSize: 4
    maxWorkers: 4
    """

MODEL_CONFIG_YAML_GPU = """
    #frontend settings
    # TorchServe frontend parameters
    minWorkers: 1
    batchSize: 4
    maxWorkers: 4
    deviceType: "gpu"
    """

MODEL_CONFIG_YAML_CPU = """
    #frontend settings
    # TorchServe frontend parameters
    minWorkers: 1
    batchSize: 4
    maxWorkers: 4
    deviceType: "cpu"
    """


@pytest.fixture(scope="module")
def model_name():
    yield "mnist"


@pytest.fixture(scope="module")
def work_dir(tmp_path_factory, model_name):
    return Path(tmp_path_factory.mktemp(model_name))


@pytest.fixture(scope="module")
def model_config_name(request):
    def get_config(param):
        if param == "cpu":
            return MODEL_CONFIG_YAML_CPU
        elif param == "gpu":
            return MODEL_CONFIG_YAML_GPU
        else:
            return MODEL_CONFIG_YAML

    return get_config(request.param)


@pytest.fixture(scope="module", name="mar_file_path")
def create_mar_file(work_dir, model_archiver, model_name, model_config_name):
    mar_file_path = work_dir.joinpath(model_name + ".mar")

    model_config_yaml_file = work_dir / "model_config.yaml"
    model_config_yaml_file.write_text(model_config_name)

    model_py_file = work_dir / "model.py"

    model_py_file.write_text(mnist_scriptes_py)

    handler_py_file = work_dir / "handler.py"
    handler_py_file.write_text(HANDLER_PY)

    config = ModelArchiverConfig(
        model_name=model_name,
        version="1.0",
        serialized_file=None,
        model_file=mnist_scriptes_py,  # model_py_file.as_posix(),
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


@pytest.mark.skipif(platform.machine() != "arm64", reason="Skip on Mac M1")
@pytest.mark.parametrize("model_config_name", ["gpu"], indirect=True)
def test_m1_device(model_name, model_config_name):
    response = requests.get(f"http://localhost:8081/models/{model_name}")
    assert response.status_code == 200, "Describe Failed"


@pytest.mark.skipif(platform.machine() != "arm64", reason="Skip on Mac M1")
@pytest.mark.parametrize("model_config_name", ["cpu"], indirect=True)
def test_m1_device_cpu(model_name, model_config_name):
    response = requests.get(f"http://localhost:8081/models/{model_name}")
    assert response.status_code == 404, "Describe Worked"


@pytest.mark.skipif(platform.machine() != "arm64", reason="Skip on Mac M1")
@pytest.mark.parametrize("model_config_name", ["default"], indirect=True)
def test_m1_device_default(model_name, model_config_name):
    response = requests.get(f"http://localhost:8081/models/{model_name}")
    assert response.status_code == 200, "Describe Failed"
