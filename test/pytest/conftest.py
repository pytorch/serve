import importlib
import os
import sys

import pytest

CURR_FILE_PATH = os.path.dirname(os.path.realpath(__file__))
REPO_ROOT_DIR = os.path.normpath(os.path.join(CURR_FILE_PATH, "..", ".."))


@pytest.fixture(scope="module")
def model_archiver():
    loader = importlib.machinery.SourceFileLoader(
        "archiver",
        os.path.join(
            REPO_ROOT_DIR, "model-archiver", "model_archiver", "model_packaging.py"
        ),
    )
    spec = importlib.util.spec_from_loader("archiver", loader)
    archiver = importlib.util.module_from_spec(spec)

    sys.modules["archiver"] = archiver

    loader.exec_module(archiver)

    yield archiver

    del sys.modules["archiver"]
