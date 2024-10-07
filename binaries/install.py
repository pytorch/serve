import os
import subprocess
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
        conda_cmd = ["conda", "install", "--channel", channel_dir, "-y", "torchserve", "torch-model-archiver"]
        print(f"## In directory: {os.getcwd()} | Executing command: {' '.join(conda_cmd)}")
        
        try:
            subprocess.run(conda_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            sys.exit("## Torchserve/Model archiver Installation Failed!")

    else:
        print("## Using pip to install torchserve and torch-model-archiver")
        ts_wheel = glob.glob(os.path.join(REPO_ROOT, "dist", "*.whl"))[0]
        ma_wheel = glob.glob(os.path.join(REPO_ROOT, "model-archiver", "dist", "*.whl"))[0]
        pip_cmd = ["pip", "install", ts_wheel, ma_wheel]
        print(f"## In directory: {os.getcwd()} | Executing command: {' '.join(pip_cmd)}")

        try:
            subprocess.run(pip_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            sys.exit("## Torchserve/Model archiver Installation Failed!")


if __name__ == "__main__":
    install()
