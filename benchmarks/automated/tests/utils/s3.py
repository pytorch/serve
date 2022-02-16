import os
import time
import re
from inspect import signature
import boto3

from retrying import retry
from fabric2 import Connection
from botocore.config import Config
from botocore.exceptions import ClientError

from invoke import run
from invoke.context import Context

from . import DEFAULT_REGION, IAM_INSTANCE_PROFILE, AMI_ID, LOGGER, S3_BUCKET_BENCHMARK_ARTIFACTS


def upload_folder_to_s3(folder_path, s3_uri, connection=None):
    """
    Upload test-related artifacts to unique s3 location.
    Allows each test to have a unique remote location for test scripts and files.
    :param folder_path: path of the folder that must be uploaded
    :param s3_uri: desired s3 path for the folder to reside in
    :param connection: if provided, s3 command will be run on the ec2 instance
    :return: True if uploaded succeeded else False
    """

    if connection:
        run_out = connection.run(f"aws s3 cp --recursive {folder_path}/ {s3_uri}/")
    else:
        run_out = run(f"aws s3 cp --recursive {folder_path}/ {s3_uri}/")

    return run_out.return_code


def download_folder_from_s3(s3_uri, folder_path, connection=None):
    """
    Download test-related artifacts from an s3 location. Assumes the s3 uri and folder_path exist.
    :param s3_uri: The s3 location from which to download the folder
    :param folder_path: the folder path to which download from the s3 bucket
    :param connection: if provided, the s3 command will be run on the ec2 instance
    """
    if connection:
        run_out = connection.run(f"aws s3 cp --recursive {s3_uri}/ {folder_path}/")
    else:
        run_out = run(f"aws s3 cp --recursive {s3_uri}/ {folder_path}/")

    return run_out.return_code


def delete_folder_from_s3(s3_folder, connection=None):
    """
    Delete s3 bucket data related to current test
    :param s3_folder: S3 URI for test artifacts to be removed
    :param connection: if provided, the s3 command will be run on the ec2 instance
    :return: <bool> True/False based on success/failure of removal
    """
    if connection:
        run_out = connection.run(f"aws s3 rm --recursive {s3_folder}")
    else:
        run_out = run(f"aws s3 rm --recursive {s3_folder}")

    return run_out.return_code


class ArtifactsHandler(object):
    @staticmethod
    def upload_torchserve_folder_to_instance(ec2_connection, s3_temp_folder):
        """
        :param ec2_connection: A fabric connection that is used to upload the parent 'serve' folder to the instance's root folder
        :return None
        """
        torch_serve_directory = os.path.abspath(os.path.join(__file__, "../../../../../"))

        s3_uri = os.path.join(S3_BUCKET_BENCHMARK_ARTIFACTS, s3_temp_folder)

        upload_return = upload_folder_to_s3(torch_serve_directory, s3_uri)

        # Note: assumes that an ubuntu (DLAMI) instance is being used.
        mkdir_return = ec2_connection.run("mkdir -p /home/ubuntu/serve")

        download_return = download_folder_from_s3(s3_uri, "/home/ubuntu/serve", connection=ec2_connection)

        assert not all(
            [upload_return, mkdir_return.return_code, download_return]
        ), f"Error uploading 'serve' folder to ec2_instance."

    @staticmethod
    def cleanup_temp_s3_folder(s3_temp_folder):
        """
        :param s3_temp_folder: last key of the s3 bucket whose folder needs to be cleaned up
        :return None
        """
        s3_uri = os.path.join(S3_BUCKET_BENCHMARK_ARTIFACTS, s3_temp_folder)

        delete_folder_from_s3(s3_uri)
