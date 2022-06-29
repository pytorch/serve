#! /usr/env/bin
import argparse
import glob
import os
import sys

# The following environment variables are expected to be populated in the shell environment
PYPI_USERNAME_ENV_VARIABLE = "TWINE_USERNAME"
PYPI_PASSWORD_ENV_VARIABLE = "TWINE_PASSWORD"

CONDA_TOKEN_ENV_VARIABLE = "CONDA_TOKEN"
CONDA_USER = "torchserve-staging"

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

CONDA_PACKAGES_PATH = os.path.join(REPO_ROOT, "binaries", "conda", "output")

TS_WHEEL_PATH = glob.glob(os.path.join(REPO_ROOT, "dist"))[0]
MA_WHEEL_PATH = glob.glob(os.path.join(REPO_ROOT, "model-archiver", "dist"))[0]
WA_WHEEL_PATH = glob.glob(os.path.join(REPO_ROOT, "workflow-archiver", "dist"))[0]


PACKAGES = ["torchserve", "model-archiver", "workflow-archiver"]


def upload_pypi_packages(pypi_token=None, test_pypi=False):
    """
    Takes a list of path values and uploads them to pypi using twine, using token stored in environment variable
    optionally pypi_token can be passed as an argument
    """
    os.system(f"pip3 install twine -q")

    # Note: TWINE_USERNAME and TWINE_PASSWORD are expected to be set in the environment
    for dist_path in [TS_WHEEL_PATH, MA_WHEEL_PATH, WA_WHEEL_PATH]:
        if test_pypi:
            exit_code = os.system(
                f"twine upload --username __token__ --password {pypi_token} {dist_path}/* --repository-url https://test.pypi.org/legacy/ --verbose"
            )
        else:
            exit_code = os.system(
                f"set -ex ; twine upload --username __token__ --password {pypi_token} {dist_path}/* --verbose"
            )
        if exit_code != 0:
            sys.exit(f"twine upload for path {dist_path} failed")

    if test_pypi:
        print(
            f"All packages uploaded to test.pypi.org successfully. Please install package as 'pip install -i https://test.pypi.org/simple/ <package-name>'"
        )


def upload_conda_packages():
    """
    Takes a list of path values and uploads them to anaconda.org using conda upload, using token stored in environment variable
    """

    # Identify *.tar.bz2 files to upload
    anaconda_token = os.getenv(CONDA_TOKEN_ENV_VARIABLE)

    for root, _, files in os.walk(CONDA_PACKAGES_PATH):
        for name in files:
            file_path = os.path.join(root, name)
            print(file_path)
            if any(word in file_path for word in PACKAGES) and file_path.endswith(
                "tar.bz2"
            ):
                print(f"Uploading to anaconda package: {name}")
                anaconda_upload_command = f"anaconda -t {anaconda_token} upload --user {CONDA_USER} {file_path} --force"
                print(f"cmd={anaconda_upload_command}")

                exit_code = os.system(anaconda_upload_command)

                if exit_code != 0:
                    print(f"Anaconda package upload failed for pacakge {name}")
                    return exit_code

    print(
        f"All packages uploaded to anaconda successfully under the channel {CONDA_USER}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Upload anaconda and pypi packages for torchserve and torch-model-archiver"
    )
    parser.add_argument(
        "--upload-conda-packages",
        action="store_true",
        required=False,
        help="Specify whether to upload conda packages",
    )
    parser.add_argument(
        "--upload-pypi-packages",
        action="store_true",
        required=False,
        help="Specify whether to upload pypi packages",
    )
    parser.add_argument(
        "--pypi-token",
        action="store",
        type=str,
        required=False,
        help="PyPI token for uploading binaries",
    )
    parser.add_argument(
        "--test-pypi",
        action="store_true",
        required=False,
        help="Specify whether to upload to test PyPI",
    )
    args = parser.parse_args()

    if args.upload_conda_packages:
        upload_conda_packages()

    if args.upload_pypi_packages:
        upload_pypi_packages(args.pypi_token, args.test_pypi)

    if any([args.upload_conda_packages, args.upload_pypi_packages]):
        print(f"Upload script complete")
    else:
        print(f"No packages uploaded")
