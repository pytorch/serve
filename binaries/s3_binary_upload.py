import argparse
import glob
import logging
import os
import subprocess
import sys

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
LOGGER.addHandler(logging.StreamHandler(sys.stderr))


REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

TS_WHEEL_PATH = glob.glob(os.path.join(REPO_ROOT, "dist"))[0]
MA_WHEEL_PATH = glob.glob(os.path.join(REPO_ROOT, "model-archiver", "dist"))[0]
WA_WHEEL_PATH = glob.glob(os.path.join(REPO_ROOT, "workflow-archiver", "dist"))[0]


class S3BinaryUploader:
    def __init__(self, s3_bucket: str, is_dryrun: bool = False, is_nightly: bool = False):
        """
        Initialize the uploader with s3_bucket, and s3_backup_bucket
        """
        self.s3_bucket = s3_bucket
        self.dryrun = is_dryrun
        if self.dryrun:
            self.s3_command = "aws s3 cp --recursive --dryrun"
        else:
            self.s3_command = "aws s3 cp --recursive"

        self.channel = "nightly" if is_nightly else ""

    def s3_upload_local_folder(self, local_folder_path: str):
        """
        Uploads the  *.whl files provided in a local folder to s3 bucket
        :params
        local_folder_path: str: path of the folder that needs to be uploaded
        """
        LOGGER.info(f"Uploading *.whl files from folder: {local_folder_path}")
        s3_command = f"{self.s3_command} --exclude '*' --include '*.whl' {local_folder_path} {self.s3_bucket.rstrip('/')}/whl/{self.channel}"

        try:
            ret_code = subprocess.run(
                s3_command, check=True, stdout=subprocess.PIPE, universal_newlines=True, shell=True
            )
        except subprocess.CalledProcessError as e:
            LOGGER.info(f"S3 upload command failed: {s3_command}. Exception: {e}")

        LOGGER.info(f"S3 upload using command: {s3_command}")

    def s3_upload_default_binaries(self):
        """
        Uploads the *.whl files provided in the standard directory structure of the pytorch 'serve' directory,
        assuming that the *.whl files are available in the 'dist' folder of the 
        """
        for local_folder_path in [TS_WHEEL_PATH, MA_WHEEL_PATH, WA_WHEEL_PATH]:
            self.s3_upload_local_folder(local_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="argument parser for s3_binary_upload.py")
    parser.add_argument(
        "--s3-bucket", required=True, help="Specify the s3 bucket to which the binaries will be uploaded"
    )
    parser.add_argument(
        "--dry-run",
        required=False,
        action="store_true",
        default=False,
        help="Specify if you just want to dry-run the upload",
    )
    parser.add_argument(
        "--nightly",
        required=False,
        action="store_true",
        default=False,
        help="Specify if you wnat to upload the binaries to the 'nightly' subfolder",
    )
    parser.add_argument(
        "--local-binaries-path",
        required=False,
        default=None,
        help="Specify a path to a folder with *.whl files, else default path to *.whl files will be used",
    )

    args = parser.parse_args()

    s3BinaryUploader = S3BinaryUploader(args.s3_bucket, args.dry_run, args.nightly)

    if args.local_binaries_path:
        s3BinaryUploader.s3_upload_local_folder(args.local_binaries_path)
    else:
        s3BinaryUploader.s3_upload_default_binaries()

    args = parser.parse_args()

