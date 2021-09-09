

import os
import workflow_archiver


def test_workflow_export_tool_version():
    """
    Test the model archive version
    :return:
    """
    with open(os.path.join('workflow_archiver', 'version.txt')) as f:
        __version__ = f.readline().strip()

    assert __version__ == str(workflow_archiver.__version__), "Versions do not match"
