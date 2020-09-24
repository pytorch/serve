import os
import glob
import sys

BASE_DIR = os.getcwd()

IS_CONDA_ENV = True if os.system("conda") == 0 else False
if IS_CONDA_ENV:
    print("Using conda to install torchserve and torch-model-archiver")
    channel_dir = os.path.abspath(os.path.join(BASE_DIR, "binaries", "conda", "output"))
    conda_cmd = f"conda install --channel {channel_dir} -y torchserve torch-model-archiver"
    INSTALL_EXIT_CODE = os.system(conda_cmd)
else:
    print("Using pip to install torchserve and torch-model-archiver")
    TS_WHEEL = glob.glob(os.path.join(BASE_DIR, "dist", "*.whl"))[0]
    MA_WHEEL = glob.glob(os.path.join(BASE_DIR, "model-archiver", "dist", "*.whl"))[0]
    pip_cmd = f"pip install {TS_WHEEL} {MA_WHEEL}"
    INSTALL_EXIT_CODE = os.system(pip_cmd)

if INSTALL_EXIT_CODE != 0 :
    sys.exit("Installation Failed")