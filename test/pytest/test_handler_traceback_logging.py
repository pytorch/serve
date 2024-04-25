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
from typing import List, Dict, Any
from ts.context import Context


class FailingModel(object):
    def __init__(self) -> None:
        pass

    def initialize(self, context: Context) -> None:
        # Deliberate bug in handler with nested calls to test traceback logging
        self.call1()

    def handle(self, data: List[Dict[str, Any]], context: Context):
        return None

    def call1(self):
        self.call2()

    def call2(self):
        self.call3()

    def call3(self):
        self.call4()

    def call4(self):
        self.call5()

    def call5(self):
        assert False
"""

MODEL_CONFIG_YAML = """
maxRetryTimeoutInSec: 300
"""

CONFIG_PROPERTIES = """
default_response_timeout=120
"""


@pytest.fixture(scope="module")
def model_name():
    yield "test_model"


@pytest.fixture(scope="module")
def work_dir(tmp_path_factory, model_name):
    return Path(tmp_path_factory.mktemp(model_name))


@pytest.fixture(scope="module")
def torchserve(model_store, work_dir):
    test_utils.torchserve_cleanup()

    config_properties_file = work_dir / "config.properties"
    config_properties_file.write_text(CONFIG_PROPERTIES)

    pipe = test_utils.start_torchserve(
        model_store=model_store,
        no_config_snapshots=True,
        gen_mar=False,
        snapshot_file=config_properties_file.as_posix(),
    )

    yield pipe

    test_utils.torchserve_cleanup()


@pytest.fixture(scope="module", name="mar_file_path")
def create_mar_file(work_dir, model_archiver, model_name):
    mar_file_path = work_dir.joinpath(model_name + ".mar")

    model_py_file = work_dir / "model.py"
    model_py_file.write_text(MODEL_PY)

    model_config_yaml_file = work_dir / "model_config.yaml"
    model_config_yaml_file.write_text(MODEL_CONFIG_YAML)

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
        ("synchronous", "false"),
        ("batch_size", "1"),
    )

    test_utils.reg_resp = test_utils.register_model_with_params(params)

    yield model_name, torchserve

    test_utils.unregister_model(model_name)


@pytest.mark.timeout(120)
def test_handler_traceback_logging(model_name):
    """
    Full circle test with torchserve
    """

    model_name, pipe = model_name

    traceback = [
        "Traceback (most recent call last):",
        "line 12, in initialize",
        "self.call1()",
        "line 18, in call1",
        "self.call2()",
        "line 21, in call2",
        "self.call3()",
        "line 24, in call3",
        "self.call4()",
        "line 27, in call4",
        "self.call5()",
        "line 30, in call5",
        "assert False",
        "AssertionError",
    ]

    # Test traceback logging for first attempt and three retries to start worker
    for _ in range(4):
        logs = []
        while True:
            logs.append(pipe.get())
            if "AssertionError" in logs[-1]:
                break

        for line in traceback:
            assert any(line in log for log in logs)
