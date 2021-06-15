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


def build(is_staging=False):
    
    ts_setup_file = glob.glob(os.path.join(REPO_ROOT, "setup.py"))[0]
    ma_setup_file = glob.glob(os.path.join(REPO_ROOT, "model-archiver","setup.py"))[0]
    wa_setup_file = glob.glob(os.path.join(REPO_ROOT, "workflow-archiver","setup.py"))[0]
   
    if is_staging:
        f_ts = Path(ts_setup_file)
        f_ts.write_text(f_ts.read_text().replace(f"name='torchserve'", f"name='torchserve-staging'"))

        f_ma = Path(ma_setup_file)
        f_ma.write_text(f_ma.read_text().replace(f"name='torch-model-archiver'", f"name='torch-model-archiver-staging'"))

        f_wa = Path(wa_setup_file)
        f_wa.write_text(f_wa.read_text().replace(f"name='torch-workflow-archiver'", f"name='torch-workflow-archiver-staging'"))

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
    
    # Build model archiver wheel
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
    
    # Revert setup.py changes
    if is_staging:
        f_ts = Path(ts_setup_file)
        f_ts.write_text(f_ts.read_text().replace(f"name='torchserve-staging'", f"name='torchserve'"))

        f_ma = Path(ma_setup_file)
        f_ma.write_text(f_ma.read_text().replace(f"name='torch-model-archiver-staging'", f"name='torch-model-archiver'"))

        f_wa = Path(wa_setup_file)
        f_wa.write_text(f_wa.read_text().replace(f"name='torch-workflow-archiver-staging'", f"name='torch-workflow-archiver'"))

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
    parser = argparse.ArgumentParser(description="argument parser for build.py")
    parser.add_argument("--staging", default=False, required=False, action="store_true", help="Specify if you want to build packages only for staging/testing")
    
    args = parser.parse_args()
    
    build(is_staging=args.staging)
