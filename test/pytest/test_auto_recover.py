import json
import platform
import shutil
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests
import test_utils

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
import time

from typing import List, Dict, Any, Tuple
from ts.context import Context


class FailingModel(object):
    def __init__(self) -> None:
        pass

    def initialize(self, context: Context) -> None:
        print(f"[xxx] Model initialization ... !!")
        self.initialized = True
        print(f"[xxx] Model initialization ... DONE !!")

    def handle(self, data: List[Dict[str, Any]], context: Context):
        self.context = context

        output = list()
        for idx, row in enumerate(data):
            # run
            print(f"[xxx] run ... !!")
            time.sleep(5)
            print(f"[xxx] run ... DONE !!")
            output.append(f"sample output {idx}")
        return output
"""

CONFIG_PROPERTIES = """
default_response_timeout=2
"""


@pytest.fixture(scope="module")
def model_name():
    yield "tp_model"


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

    handler_py_file = work_dir / "handler.py"
    handler_py_file.write_text(HANDLER_PY)

    args = Namespace(
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
        config_file=None,
    )

    mock = MagicMock()
    mock.parse_args = MagicMock(return_value=args)
    with patch("archiver.ArgParser.export_model_args_parser", return_value=mock):
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
        ("synchronous", "true"),
        ("batch_size", "1"),
    )

    test_utils.reg_resp = test_utils.register_model_with_params(params)

    yield model_name, torchserve

    test_utils.unregister_model(model_name)


@pytest.mark.skipif(
    platform.system() != "Linux", reason="Skipping test on non-Linux system"
)
def test_tp_inference(model_name):
    """
    Full circle test with torchserve
    """

    model_name, pipe = model_name

    response = requests.post(
        url=f"http://localhost:8080/predictions/{model_name}", data=json.dumps(42)
    )
    assert response.status_code == 500

    logs = []
    for _ in range(100):
        logs.append(pipe.get())
        if "Auto recovery succeeded, reset recoveryStartTS" in logs[-1]:
            break

    assert any("Model initialization ... DONE" in l for l in logs)
    assert any("Number or consecutive unsuccessful inference 1" in l for l in logs)
    assert any("Worker disconnected" in l for l in logs)
    assert any("Retry worker" in l for l in logs)
    assert any("Auto recovery start timestamp" in l for l in logs)
    assert not any("Auto recovery failed again" in l for l in logs)
