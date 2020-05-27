

import os
import ts


def test_ts_version():
    with open(os.path.join("ts", "version.txt")) as f:
        __version__ = f.readline().strip()

    assert __version__ == str(ts.__version__), "Versions don't match"
