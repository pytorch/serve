

import os
import model_archiver


def test_model_export_tool_version():
    """
    Test the model archive version
    :return:
    """
    with open(os.path.join('model_archiver', 'version.txt')) as f:
        __version__ = f.readline().strip()

    assert __version__ == str(model_archiver.__version__), "Versions do not match"
