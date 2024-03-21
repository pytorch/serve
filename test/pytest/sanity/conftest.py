import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[3]


MAR_CONFIG = REPO_ROOT.joinpath("ts_scripts", "mar_config.json")


@pytest.fixture(name="gen_models", scope="module")
def load_gen_models() -> dict:
    with open(MAR_CONFIG) as f:
        models = json.load(f)
    models = {m["model_name"]: m for m in models}
    return models


@pytest.fixture(scope="module")
def ts_scripts_path():
    sys.path.append(REPO_ROOT.as_posix())

    yield

    sys.path.pop()
