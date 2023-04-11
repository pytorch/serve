import shutil
import sys
from pathlib import Path

import pytest

from .models.base_model import save_pt_file
from .test_utils.mock_context import MockContext


@pytest.fixture()
def base_model_dir(tmp_path_factory):
    model_dir = tmp_path_factory.mktemp("base_model_dir")

    shutil.copyfile(
        Path(__file__).parents[0] / "models" / "base_model.py", model_dir / "model.py"
    )

    save_pt_file(model_dir.joinpath("model.pt").as_posix())

    sys.path.append(model_dir.as_posix())
    yield model_dir
    sys.path.pop()


@pytest.fixture()
def base_model_context(base_model_dir):

    context = MockContext(
        model_name="mnist",
        model_dir=base_model_dir.as_posix(),
        model_file="model.py",
    )
    yield context
