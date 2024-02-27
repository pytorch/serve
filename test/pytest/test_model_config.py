import shutil
from pathlib import Path
from unittest.mock import patch

import pytest
import test_utils
from model_archiver import ModelArchiverConfig

CURR_FILE_PATH = Path(__file__).parent
REPO_ROOT_DIR = CURR_FILE_PATH.parent.parent

MODEL_PY = """
import torch
import torch.nn as nn

class Foo(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
"""

HANDLER_PY = """
import os
import torch
from ts.torch_handler.base_handler import BaseHandler

class FooHandler(BaseHandler):
    def initialize(self, ctx):
        super().initialize(ctx)

    def preprocess(self, data):
        return torch.as_tensor(int(data[0].get('body').decode('utf-8')), device=self.device)

    def postprocess(self, x):
        return [x.item()]
"""

MODEL_CONFIG_YAML = f"""
#frontend settings
# TorchServe frontend parameters
minWorkers: 1
maxWorkers: 4
maxBatchDelay: 100
batchSize: 4
"""


@pytest.fixture(scope="module")
def model_name():
    yield "foo"


@pytest.fixture(scope="module")
def work_dir(tmp_path_factory, model_name):
    return Path(tmp_path_factory.mktemp(model_name))


@pytest.fixture(scope="module", name="mar_file_path")
def create_mar_file(work_dir, model_archiver, model_name):
    mar_file_path = work_dir.joinpath(model_name + ".mar")

    model_config_yaml_file = work_dir / "model_config.yaml"
    model_config_yaml_file.write_text(MODEL_CONFIG_YAML)

    model_py_file = work_dir / "model.py"
    model_py_file.write_text(MODEL_PY)

    handler_py_file = work_dir / "handler.py"
    handler_py_file.write_text(HANDLER_PY)

    config = ModelArchiverConfig(
        model_name=model_name,
        version="1.0",
        serialized_file=None,
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


@pytest.fixture(scope="module", name="base_model_name")
def register_model(mar_file_path, model_store, torchserve):
    shutil.copy(mar_file_path, model_store)

    file_name = Path(mar_file_path).name

    base_model_name = Path(file_name).stem

    params = (
        ("model_name", f"{base_model_name}_b4"),
        ("url", file_name),
        ("initial_workers", "2"),
        ("synchronous", "true"),
    )

    test_utils.reg_resp = test_utils.register_model_with_params(params)

    params = (
        ("model_name", f"{base_model_name}_b2"),
        ("url", file_name),
        ("initial_workers", "2"),
        ("synchronous", "true"),
        ("batch_size", "2"),
    )

    test_utils.reg_resp = test_utils.register_model_with_params(params)
    yield base_model_name


def test_register_model_with_batch_size(base_model_name):
    describe_resp = test_utils.describe_model(f"{base_model_name}_b2", "1.0")

    assert describe_resp[0]["batchSize"] == 2


def test_register_model_without_batch_size(base_model_name):
    describe_resp = test_utils.describe_model(f"{base_model_name}_b4", "1.0")

    assert describe_resp[0]["batchSize"] == 4
