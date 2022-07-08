import argparse
import glob
import os
import sys

# To help discover local modules
REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(REPO_ROOT)


from binaries.conda.build_packages import (
    conda_build,
    install_conda_build,
    install_miniconda,
)
from ts_scripts.utils import is_conda_build_env, is_conda_env


def build_dist_whl(nightly=False):
    """
    Function to build the wheel files for torchserve, model-archiver and workflow-archiver
    nightly is an optional argument to build nightlies
    """

    binaries = ["torchserve", "model-archiver", "workflow-archiver"]
    if nightly:
        print(
            "## Started torchserve, model-archiver and workflow-archiver nightly build"
        )
        create_wheel_cmd = "python setup.py "
    else:
        print("## Started torchserve, model-archiver and workflow-archiver build")
        create_wheel_cmd = "python setup.py bdist_wheel --release --universal"

    for binary in binaries:
        if "serve" in binary:
            cur_dir = REPO_ROOT
        else:
            cur_dir = REPO_ROOT + "/" + binary
        os.chdir(cur_dir)

        if nightly:
            cur_wheel_cmd = (
                create_wheel_cmd
                + "--override-name "
                + binary
                + "-nightly"
                + " bdist_wheel"
            )
        else:
            cur_wheel_cmd = create_wheel_cmd

        # Build wheel
        print(f"## In directory: {os.getcwd()} | Executing command: {cur_wheel_cmd}")
        build_exit_code = os.system(cur_wheel_cmd)

        # If any one of the steps fail, exit with error
        if build_exit_code != 0:
            sys.exit("## {} build Failed !".format(binary))


def build(args):

    # Build dist wheel files
    build_dist_whl(args.nightly)

    os.chdir(REPO_ROOT)

    ts_wheel_path = glob.glob(os.path.join(REPO_ROOT, "dist", "*.whl"))[0]
    ma_wheel_path = glob.glob(
        os.path.join(REPO_ROOT, "model-archiver", "dist", "*.whl")
    )[0]
    wa_wheel_path = glob.glob(
        os.path.join(REPO_ROOT, "workflow-archiver", "dist", "*.whl")
    )[0]
    print(f"## TorchServe wheel location: {ts_wheel_path}")
    print(f"## Model archiver wheel location: {ma_wheel_path}")
    print(f"## Workflow archiver wheel location: {ma_wheel_path}")

    # Build TS & MA on Conda if available
    conda_build_exit_code = 0
    if not is_conda_env():
        install_miniconda()

    if not is_conda_build_env():
        install_conda_build()

    conda_build_exit_code = conda_build(
        ts_wheel_path,
        ma_wheel_path,
        wa_wheel_path,
        args.nightly and args.upload_conda_packages,
    )

    # If conda build fails, exit with error
    if conda_build_exit_code != 0:
        sys.exit("## Conda Build Failed !")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build for torchserve, torch-model-archiver and torch-workflow-archiver"
    )
    parser.add_argument(
        "--upload-conda-packages",
        action="store_true",
        required=False,
        help="Specify whether to upload conda packages",
    )
    parser.add_argument(
        "--nightly",
        action="store_true",
        required=False,
        help="specify nightly is being built",
    )
    args = parser.parse_args()

    build(args)
