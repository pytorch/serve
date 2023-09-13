import argparse
import os
import sys

# To help discover local modules
REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(REPO_ROOT)

import datetime

from ts_scripts import marsgen as mg
from ts_scripts.api_utils import test_api
from ts_scripts.install_from_src import install_from_src
from ts_scripts.regression_utils import test_regression
from ts_scripts.utils import check_python_version, try_and_handle


def regression_tests(binaries, pypi, conda, nightly):
    now = datetime.datetime.now()
    print("Current date and time : " + now.strftime("%Y-%m-%d %H:%M:%S"))

    check_python_version()

    if binaries:
        if pypi:
            if nightly:
                cmd = f"pip install torchserve-nightly torch-model-archiver-nightly torch-workflow-archiver-nightly"
            else:
                cmd = f"pip install torchserve torch-model-archiver torch-workflow-archiver"
        elif conda:
            if nightly:
                cmd = f"conda install -c pytorch-nightly torchserve torch-model-archiver torch-workflow-archiver"
            else:
                cmd = f"conda install -c pytorch torchserve torch-model-archiver torch-workflow-archiver"
        try_and_handle(cmd, False)
        print(f"## In directory: {os.getcwd()}; Executing command: {cmd}")
    else:
        # Install from source
        install_from_src()

    # Generate mar file
    mg.generate_mars()

    # Run newman api tests
    test_api(
        "all"
    )  # "all" > management, inference, increased_timeout_inference, https collections

    # Run regression tests
    test_regression()

    # delete mar_gen_dir
    mg.delete_model_store_gen_dir()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Regression tests for TorchServe")
    parser.add_argument(
        "--nightly",
        action="store_true",
        required=False,
        help="Run regression tests using nightly binaries",
    )

    parser.add_argument(
        "--binaries",
        action="store_true",
        required=False,
        help="Run regression tests using binaries",
    )

    parser.add_argument(
        "--pypi",
        action="store_true",
        required=False,
        help="Run regression tests using pypi",
    )

    parser.add_argument(
        "--conda",
        action="store_true",
        required=False,
        help="Run regression tests using conda",
    )

    args = parser.parse_args()

    regression_tests(args.binaries, args.pypi, args.conda, args.nightly)
