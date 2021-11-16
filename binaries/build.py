import argparse
import os
import sys
import glob

# To help discover local modules
REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(REPO_ROOT)

from pathlib import Path

from ts_scripts.utils import is_conda_env, is_conda_build_env
from binaries.conda.build_packages import conda_build, install_miniconda, install_conda_build


def build():
    
    print("## Started torchserve, model-archiver and workflow-archiver build")
    create_wheel_cmd = "python setup.py bdist_wheel --release --universal"

    # Build torchserve wheel
    print(f"## In directory: {os.getcwd()} | Executing command: {create_wheel_cmd}")
    ts_build_exit_code = os.system(create_wheel_cmd)

    # Build model archiver wheel
    os.chdir("model-archiver")
    print(f"## In directory: {os.getcwd()} | Executing command: {create_wheel_cmd}")
    ma_build_exit_code = os.system(create_wheel_cmd)

    os.chdir(REPO_ROOT)
    
    # Build workflow archiver wheel
    os.chdir("workflow-archiver")
    print(f"## In directory: {os.getcwd()} | Executing command: {create_wheel_cmd}")
    wa_build_exit_code = os.system(create_wheel_cmd)

    os.chdir(REPO_ROOT)

    ts_wheel_path = glob.glob(os.path.join(REPO_ROOT, "dist", "*.whl"))[0]
    ma_wheel_path = glob.glob(os.path.join(REPO_ROOT, "model-archiver", "dist", "*.whl"))[0]
    wa_wheel_path = glob.glob(os.path.join(REPO_ROOT, "workflow-archiver", "dist", "*.whl"))[0]
    print(f"## TorchServe wheel location: {ts_wheel_path}")
    print(f"## Model archiver wheel location: {ma_wheel_path}")
    print(f"## Workflow archiver wheel location: {ma_wheel_path}")

    # Build TS & MA on Conda if available
    conda_build_exit_code = 0
    if not is_conda_env():
        install_miniconda()
    
    if not is_conda_build_env():
        install_conda_build()

    conda_build_exit_code = conda_build(ts_wheel_path, ma_wheel_path, wa_wheel_path)
    
    # If any one of the steps fail, exit with error
    if ts_build_exit_code != 0:
        sys.exit("## Torchserve Build Failed !")
    if ma_build_exit_code != 0:
        sys.exit("## Model archiver Build Failed !")
    if wa_build_exit_code != 0:
        sys.exit("## Workflow archiver build failed !")
    if conda_build_exit_code != 0:
        sys.exit("## Conda Build Failed !")


if __name__ == "__main__":
    
    build()
