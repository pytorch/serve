

import os
import model_archiver


def test_model_export_tool_version():
    """
    Test the model archive version
    :return:
    """
    with (open(os.path.join('model_archiver', 'version.py'))) as f:
        exec(f.read(), globals())

    assert __version__ == str(model_archiver.__version__), "Versions do not match"
