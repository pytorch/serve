from pathlib import Path

import model_archiver

MODEL_ARCHIVER_ROOT_DIR = Path(__file__).parent.parent.parent


def test_model_export_tool_version():
    """
    Test the model archive version
    :return:
    """
    with open(MODEL_ARCHIVER_ROOT_DIR.joinpath("version.txt")) as f:
        __version__ = f.readline().strip()

    assert __version__ == str(model_archiver.__version__), "Versions do not match"
