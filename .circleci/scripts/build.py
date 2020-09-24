import os
import sys
import glob

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
IS_CONDA_ENV = True if os.system("conda") == 0 else False
if IS_CONDA_ENV:
    ts_wheel_path = glob.glob(os.path.join(BASE_DIR, "dist", "*.whl"))[0]
    ma_wheel_path = glob.glob(os.path.join(BASE_DIR, "model-archiver", "dist", "*.whl"))[0]
    conda_build_ts(ts_wheel_path, ma_wheel_path)

#
# if IS_CONDA_ENV :
#     TS_WHL_PATH=os.path.join(BASE_DIR, "dist", "*.whl")
#     MA_WHL_PATH=os.path.join(BASE_DIR, "model-archiver", "dist", "*.whl")

# (
#   set -e
#   if [ -x "$(command -v conda)" ]
#   then
#       BASE_PATH=$(pwd)
#       TS_WHL_PATH=$BASE_PATH/$(ls dist/*.whl)
#       MA_WHL_PATH=$BASE_PATH/$(ls model-archiver/dist/*.whl)
#       cd binaries/conda
#       TORCHSERVE_WHEEL=$TS_WHL_PATH TORCH_MODEL_ARCHIVER_WHEEL=$MA_WHL_PATH ./build_packages.sh
#   fi
# )
# CONDA_BUILD_EXIT_CODE=$?

if any( EXIT_CODE != 0 for EXIT_CODE in [TS_BUILD_EXIT_CODE, MA_BUILD_EXIT_CODE]):
    sys.exit("Build Failed")