import os
import sys

BASE_DIR = os.getcwd()
CREATE_WHEEL_CMD = "python setup.py bdist_wheel --release --universal"

# Build torchserve wheel
TS_BUILD_EXIT_CODE = os.system(CREATE_WHEEL_CMD)

# Build model archiver wheel
os.chdir("model-archiver")
MA_BUILD_EXIT_CODE = os.system(CREATE_WHEEL_CMD)

os.chdir(BASE_DIR)

# Build TS & MA on Conda if available
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