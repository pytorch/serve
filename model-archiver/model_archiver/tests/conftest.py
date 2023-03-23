import json
from pathlib import Path

import pytest

INTEG_TEST_CONFIG_FILE = "integ_tests/configuration.json"
DEFAULT_HANDLER_CONFIG_FILE = "integ_tests/default_handler_configuration.json"

TEST_ROOT_DIR = Path(__file__).parent
MODEL_ARCHIVER_ROOT_DIR = Path(__file__).parents[2]


def make_paths_absolute(test, keys):
    def make_absolute(paths):
        if "," in paths:
            return ",".join([make_absolute(p) for p in paths.split(",")])
        return MODEL_ARCHIVER_ROOT_DIR.joinpath(paths).as_posix()

    for k in keys:
        test[k] = make_absolute(test[k])

    return test


@pytest.fixture(name="integ_tests")
def load_integ_tests():
    with open(TEST_ROOT_DIR.joinpath(INTEG_TEST_CONFIG_FILE), "r") as f:
        tests = json.loads(f.read())
    keys = (
        "model-file",
        "serialized-file",
        "handler",
        "extra-files",
    )
    return [make_paths_absolute(t, keys) for t in tests]


@pytest.fixture(name="default_handler_tests")
def load_default_handler_tests():
    with open(TEST_ROOT_DIR.joinpath(DEFAULT_HANDLER_CONFIG_FILE), "r") as f:
        default_handler_tests = json.loads(f.read())
    keys = (
        "model-file",
        "serialized-file",
        "extra-files",
    )
    default_handler_tests = [
        make_paths_absolute(t, keys) for t in default_handler_tests
    ]
    return default_handler_tests
