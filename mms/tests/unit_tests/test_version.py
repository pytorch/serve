

import os
import re

import mms


def test_mms_version():
    with open(os.path.join("mms", "version.py")) as f:
        exec(f.read(), globals())

    assert __version__ == str(mms.__version__), "Versions don't match"
