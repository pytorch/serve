import os
import sys
import glob

# To help discover local modules
REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(REPO_ROOT)

from ts_scripts.utils import is_conda_env


def install():
    if is_conda_env():
        print("## Using conda to install torchserve and torch-model-archiver")
        channel_dir = os.path.abspath(os.path.join(REPO_ROOT, "binaries", "conda", "output"))
        conda_cmd = f"conda install --channel {channel_dir} -y torchserve torch-model-archiver"
        print(f"## In directory: {os.getcwd()} | Executing command: {conda_cmd}")
        install_exit_code = os.system(conda_cmd)
    else:
        print("## Using pip to install torchserve and torch-model-archiver")
        ts_wheel = glob.glob(os.path.join(REPO_ROOT, "dist", "*.whl"))[0]
        ma_wheel = glob.glob(os.path.join(REPO_ROOT, "model-archiver", "dist", "*.whl"))[0]
        pip_cmd = f"pip install {ts_wheel} {ma_wheel}"
        print(f"## In directory: {os.getcwd()} | Executing command: {pip_cmd}")
        install_exit_code = os.system(pip_cmd)

    if install_exit_code != 0:
        sys.exit("## Torchserve \ Model archiver Installation Failed !")


if __name__ == "__main__":
    install()
