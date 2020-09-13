#!/bin/bash

# Build torchserve wheel
python setup.py bdist_wheel --release --universal
TS_BUILD_EXIT_CODE=$?

# Build model archiver wheel
cd model-archiver/
python setup.py bdist_wheel --release --universal
MA_BUILD_EXIT_CODE=$?
cd ../

# Build TS & MA on Conda if available
(
  set -e
  if [ -x "$(command -v conda)" ]
  then
      BASE_PATH=$(pwd)
      TS_WHL_PATH=$BASE_PATH/$(ls dist/*.whl)
      MA_WHL_PATH=$BASE_PATH/$(ls model-archiver/dist/*.whl)
      cd binaries/conda
      TORCHSERVE_WHEEL=$TS_WHL_PATH TORCH_MODEL_ARCHIVER_WHEEL=$MA_WHL_PATH ./build_packages.sh
  fi
)
CONDA_BUILD_EXIT_CODE=$?

# If any one of the builds fail, exit with error
if [ "$TS_BUILD_EXIT_CODE" -ne 0 ] || [ "$MA_BUILD_EXIT_CODE" -ne 0 ] || [ "$CONDA_BUILD_EXIT_CODE" -ne 0 ]
then exit 1
fi