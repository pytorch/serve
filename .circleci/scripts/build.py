import os
import sys
import glob
from scripts import install_utils

sys.path.append(os.getcwd()) # Need from conda_build_ts
from binaries.conda.build_packages import conda_build_ts

BASE_DIR = os.getcwd()
CREATE_WHEEL_CMD = "python setup.py bdist_wheel --release --universal"

# Build torchserve wheel
TS_BUILD_EXIT_CODE = os.system(CREATE_WHEEL_CMD)

# Build model archiver wheel
os.chdir("model-archiver")
MA_BUILD_EXIT_CODE = os.system(CREATE_WHEEL_CMD)

os.chdir(BASE_DIR)

# Build TS & MA on Conda if available
if install_utils.is_conda_env():
    ts_wheel_path = glob.glob(os.path.join(BASE_DIR, "dist", "*.whl"))[0]
    ma_wheel_path = glob.glob(os.path.join(BASE_DIR, "model-archiver", "dist", "*.whl"))[0]
    conda_build_ts(ts_wheel_path, ma_wheel_path)

if any( EXIT_CODE != 0 for EXIT_CODE in [TS_BUILD_EXIT_CODE, MA_BUILD_EXIT_CODE]):
    sys.exit("Build Failed")