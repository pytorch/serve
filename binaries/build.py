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


def build_dist_whl(args):
    """
    Function to build the wheel files for torchserve, model-archiver and workflow-archiver
    """
    binaries = ["torchserve", "torch-model-archiver", "torch-workflow-archiver"]
    if args.nightly:
        print(
            "## Started torchserve, model-archiver and workflow-archiver nightly build"
        )
        create_wheel_cmd = "python setup.py "
    else:
        print("## Started torchserve, model-archiver and workflow-archiver build")
        create_wheel_cmd = "python setup.py bdist_wheel --release"

    for binary in binaries:

        if "serve" in binary:
            cur_dir = REPO_ROOT
        else:
            cur_dir = os.path.join(REPO_ROOT, binary[len("torch-") :])

        os.chdir(cur_dir)

        cur_wheel_cmd = (
            create_wheel_cmd + "--override-name " + binary + "-nightly" + " bdist_wheel"
            if args.nightly
            else create_wheel_cmd
        )

        # Build wheel
        print(f"## In directory: {os.getcwd()} | Executing command: {cur_wheel_cmd}")

        if not args.dry_run:
            build_exit_code = os.system(cur_wheel_cmd)
            # If any one of the steps fail, exit with error
            if build_exit_code != 0:
                sys.exit(f"## {binary} build Failed !")


def build(args):

    # Build dist wheel files
    build_dist_whl(args)

    os.chdir(REPO_ROOT)

    if not args.dry_run:
        ts_wheel_path = glob.glob(os.path.join(REPO_ROOT, "dist", "*.whl"))[0]
        ma_wheel_path = glob.glob(
            os.path.join(REPO_ROOT, "model-archiver", "dist", "*.whl")
        )[0]
        wa_wheel_path = glob.glob(
            os.path.join(REPO_ROOT, "workflow-archiver", "dist", "*.whl")
        )[0]

    else:
        ts_wheel_path = os.path.join(REPO_ROOT, "dist", "*.whl")
        ma_wheel_path = os.path.join("model-archiver", "dist", "*.whl")
        wa_wheel_path = os.path.join("workflow-archiver", "dist", "*.whl")

    print(f"## TorchServe wheel location: {ts_wheel_path}")
    print(f"## Model archiver wheel location: {ma_wheel_path}")
    print(f"## Workflow archiver wheel location: {ma_wheel_path}")

    # Build TS & MA on Conda if available
    conda_build_exit_code = 0
    if not is_conda_env():
        install_miniconda(args.dry_run)

    if not is_conda_build_env():
        install_conda_build(args.dry_run)

    conda_build_exit_code = conda_build(
        ts_wheel_path, ma_wheel_path, wa_wheel_path, args.nightly, args.dry_run
    )

    # If conda build fails, exit with error
    if conda_build_exit_code != 0:
        sys.exit("## Conda Build Failed !")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build for torchserve, torch-model-archiver and torch-workflow-archiver"
    )
    parser.add_argument(
        "--nightly",
        action="store_true",
        required=False,
        help="specify nightly is being built",
    )

    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="dry_run will print the commands that will be run without running them",
    )

    args = parser.parse_args()

    build(args)
