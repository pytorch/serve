#!/bin/bash

if [ -x "$(command -v conda)" ]
then # Install from conda packages, if conda is available
  BASE_PATH=$(pwd)
  conda install -c file://$BASE_PATH/binaries/conda/output -y torchserve torch-model-archiver
else # Install from torchserve and torch model archiver wheel
  pip install dist/*.whl model-archiver/dist/*.whl
fi

# Exit code is as per the last line in if - else blocks, so no need of explicit handling
# Incase the code changes, Remember to handle exit code for CI builds