import os
import sys

BASE_DIR = os.getcwd()

TS_WHEEL = os.path.join(BASE_DIR, "dist", "*.whl")
MA_WHEEL = os.path.join(BASE_DIR, "model-archiver", "dist", "*.whl")

INSTALL_EXIT_CODE = os.system(f"pip install {TS_WHEEL} {MA_WHEEL}")

if any( EXIT_CODE != 0 for EXIT_CODE in [INSTALL_EXIT_CODE]):
    sys.exit("Installation Failed")

# Exit code is as per the last line in if - else blocks, so no need of explicit handling
# Incase the code changes, Remember to handle exit code for CI builds


# if [ -x "$(command -v conda)" ]
# then # Install from conda packages, if conda is available
#   BASE_PATH=$(pwd)
#   conda install -c file://$BASE_PATH/binaries/conda/output -y torchserve torch-model-archiver
# fi

# Exit code is as per the last line in if - else blocks, so no need of explicit handling
# Incase the code changes, Remember to handle exit code for CI builds