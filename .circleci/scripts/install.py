import os
import sys
import glob

sys.path.append(os.getcwd())  # To help discover local modules
from scripts import tsutils

REPO_ROOT = os.getcwd()


def install():
    if tsutils.is_conda_env():
        print("## Using conda to install torchserve and torch-model-archiver")
        channel_dir = os.path.abspath(os.path.join(REPO_ROOT, "binaries", "conda", "output"))
        conda_cmd = f"conda install --channel {channel_dir} -y torchserve torch-model-archiver"
        print(f"## Executing command: {conda_cmd}")
        install_exit_code = os.system(conda_cmd)
    else:
        print("## Using pip to install torchserve and torch-model-archiver")
        TS_WHEEL = glob.glob(os.path.join(REPO_ROOT, "dist", "*.whl"))[0]
        MA_WHEEL = glob.glob(os.path.join(REPO_ROOT, "model-archiver", "dist", "*.whl"))[0]
        pip_cmd = f"pip install {TS_WHEEL} {MA_WHEEL}"
        print(f"## Executing command: {pip_cmd}")
        install_exit_code = os.system(pip_cmd)

    if install_exit_code != 0:
        sys.exit("## Torchserve \ Model archiver Installation Failed !")


if __name__ == "__main__":
    install()
