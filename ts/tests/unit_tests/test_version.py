

import os
import ts


def test_ts_version():
    with open(os.path.join("ts", "version.py")) as f:
        exec(f.read(), globals())

    assert __version__ == str(ts.__version__), "Versions don't match"
