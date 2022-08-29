#! /usr/env/bin
import argparse
import glob
import os
import sys

# To help discover local modules
REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(REPO_ROOT)

from ts_scripts.utils import try_and_handle


def upload_pypi_packages(args, WHL_PATHS):
    """
    Takes a list of path values and uploads them to pypi using twine, using token stored in environment variable
    """
    dry_run = args.dry_run

    # Note: TWINE_USERNAME and TWINE_PASSWORD are expected to be set in the environment
    for dist_path in WHL_PATHS:
        if args.test_pypi:
            try_and_handle(
                f"twine upload {dist_path}/* --username __token__ --repository-url https://test.pypi.org/legacy/",
                dry_run,
            )
        else:
            try_and_handle(f"twine upload --username __token__ {dist_path}/*", dry_run)


# TODO: Mock some file paths to make conda dry run work
def upload_conda_packages(args, PACKAGES, CONDA_PACKAGES_PATH):
    """
    Takes a list of path values and uploads them to anaconda.org using conda upload, using token stored in environment variable
    If you'd like to upload to a staging environment make sure to pass in your personal credentials when you anaconda login instead
    of the pytorch credentials
    """
    # Set ANACONDA_API_TOKEN before calling this function
    for root, _, files in os.walk(CONDA_PACKAGES_PATH):
        for name in files:
            file_path = os.path.join(root, name)
            # Identify *.tar.bz2 files to upload
            if any(word in file_path for word in PACKAGES) and file_path.endswith(
                "tar.bz2"
            ):
                print(f"Uploading to anaconda package: {name}")
                anaconda_upload_command = f"anaconda upload {file_path} --force"
                exit_code = os.system(anaconda_upload_command)
                if exit_code != 0:
                    print(f"Anaconda package upload failed for package {name}")
                    return exit_code
    print(f"All packages uploaded to anaconda successfully")


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
        "--test-pypi",
        action="store_true",
        required=False,
        help="Specify whether to upload to test PyPI",
    )

    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="dry_run will print the commands that will be run without running them. Only works for pypi now",
    )
    args = parser.parse_args()

    PACKAGES = ["torchserve", "model-archiver", "workflow-archiver"]
    CONDA_PACKAGES_PATH = os.path.join(REPO_ROOT, "binaries", "conda", "output")

    if not args.dry_run:
        TS_WHEEL_PATH = glob.glob(os.path.join(REPO_ROOT, "dist"))[0]
        MA_WHEEL_PATH = glob.glob(os.path.join(REPO_ROOT, "model-archiver", "dist"))[0]
        WA_WHEEL_PATH = glob.glob(os.path.join(REPO_ROOT, "workflow-archiver", "dist"))[
            0
        ]
    else:
        TS_WHEEL_PATH = os.path.join(REPO_ROOT, "dist")
        MA_WHEEL_PATH = os.path.join(REPO_ROOT, "model-archiver", "dist")
        WA_WHEEL_PATH = os.path.join(REPO_ROOT, "workflow-archiver", "dist")

    WHL_PATHS = [TS_WHEEL_PATH, MA_WHEEL_PATH, WA_WHEEL_PATH]

    if args.upload_pypi_packages:
        upload_pypi_packages(args, WHL_PATHS)

    if args.upload_conda_packages:
        upload_conda_packages(args, PACKAGES, CONDA_PACKAGES_PATH)
