from __future__ import absolute_import

import json
import logging
import fcntl
import os
import re
import subprocess
import sys
import time

import boto3
import docker
import git
import pytest
import yaml
from botocore.exceptions import ClientError
from invoke import run
from invoke.context import Context

DEFAULT_REGION = "us-west-2"
IAM_INSTANCE_PROFILE = "EC2Admin"
S3_BUCKET_BENCHMARK_ARTIFACTS = "s3://torchserve-model-serving/benchmark_artifacts"

DEFAULT_DOCKER_DEV_ECR_REPO = "torchserve-benchmark"
DEFAULT_DOCKER_DEV_ECR_TAG = "dev-image"
DEFAULT_DOCKER_DEV_ECR_REPO_TAG = f"{DEFAULT_DOCKER_DEV_ECR_REPO}:{DEFAULT_DOCKER_DEV_ECR_TAG}"
ECR_REPOSITORY_URL = "{}.dkr.ecr.{}.amazonaws.com/{}"

GPU_INSTANCES = ["p2", "p3", "p4", "g2", "g3", "g4"]

# DLAMI with nVidia Driver ver. 450.119.03 (support upto CUDA 11.2), Ubuntu 18.04
# AMI_ID = "ami-064696901389beb84"
# AMI_ID = "ami-0198925303105158c", Base DLAMI 37.0 with apache2-utils installed
AMI_ID = "ami-00c5ebd9076702cbe"#, DLAMI 43.0 with apache2-utils installed

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
LOGGER.addHandler(logging.StreamHandler(sys.stderr))


class DockerImageHandler(object):
    def __init__(self, docker_tag, cuda_version=None, branch="master"):
        self.docker_tag = docker_tag
        self.cuda_version = cuda_version
        self.branch = branch

    def build_image(self):
        """
        Uses the start_build.sh script to build a docker container with the given parameters.
        """
        torch_serve_docker_directory = os.path.abspath(os.path.join(__file__, "../../../../../docker/"))
        current_working_directory = os.getcwd()
        os.chdir(torch_serve_docker_directory)
        if self.cuda_version:
            run_out = run(
                f"./build_image.sh -b {self.branch} -bt dev -g -cv {self.cuda_version} -t {DEFAULT_DOCKER_DEV_ECR_REPO}:{self.docker_tag}"
            )
        else:
            run_out = run(f"./build_image.sh -b {self.branch} -bt dev -t {DEFAULT_DOCKER_DEV_ECR_REPO}:{self.docker_tag}")

        # Switch back to original directory
        os.chdir(current_working_directory)
        LOGGER.info(f"Dev image build successful:  {DEFAULT_DOCKER_DEV_ECR_REPO}:{self.docker_tag}")

    @staticmethod
    def push_docker_image_to_ecr(
        account_id, region, docker_repo_tag=f"{DEFAULT_DOCKER_DEV_ECR_REPO_TAG}", connection=None
    ):
        """
        :param account_id: aws account id to which the dev image must be pushed
        :param region: aws region for the ecr repo
        :param docker_repo_tag: the repo:tag for the ecr image
        :param connection: if provided, runs the command on ec2 instance
        """

        ecr_uri = ECR_REPOSITORY_URL.format(account_id, region, docker_repo_tag)

        run(f"docker tag {docker_repo_tag} {ecr_uri}")

        ecr_login_command = f"aws ecr get-login-password --region {region} | docker login --username AWS --password-stdin {account_id}.dkr.ecr.{region}.amazonaws.com"

        if connection:
            run_out = connection.run(ecr_login_command)
        else:
            run_out = run(ecr_login_command)
        assert run_out.return_code == 0, f"ECR login failed when pushing dev image to ECR"

        if connection:
            run_out = connection.run(f"docker push {ecr_uri}")
        else:
            run_out = run(f"docker push {ecr_uri}")

        assert run_out.return_code == 0, f"ECR docker push failed"

        LOGGER.info(f"Dev image push to ECR successful.")

    @staticmethod
    def pull_docker_image_from_ecr(
        account_id, region, docker_repo_tag=f"{DEFAULT_DOCKER_DEV_ECR_REPO_TAG}", connection=None
    ):
        """
        :param account_id: aws account id from which the dev image must be pulled
        :param region: aws region for the ecr repo
        :param docker_repo_tag: the repo:tag for the ecr image
        :param connection: if provided, runs the command on ec2 instance
        """
        ecr_uri = ECR_REPOSITORY_URL.format(account_id, region, docker_repo_tag)

        ecr_login_command = f"aws ecr get-login-password --region {region} | docker login --username AWS --password-stdin {account_id}.dkr.ecr.{region}.amazonaws.com"

        if connection:
            run_out = connection.run(ecr_login_command)
        else:
            run_out = run(ecr_login_command)
        assert run_out.return_code == 0, f"ECR login failed when pushing dev image to ECR"

        if connection:
            run_out = connection.run(f"docker pull {ecr_uri}")

            # Re-tag to make referencing easier for functions
            connection.run(f"docker tag {ecr_uri} {docker_repo_tag}")
        else:
            run_out = run(f"docker pull {ecr_uri}")

            connection.run(f"docker tag {ecr_uri} {docker_repo_tag}")

        assert run_out.return_code == 0, f"ECR docker push failed"

        LOGGER.info(f"Dev image pull from ECR successful.")


class YamlHandler(object):

    valid_mode_keys = ["eager_mode", "scripted_mode", "kf_serving_mode"]

    mandatory_config_keys = [
        "backend_profiling",
        "batch_delay",
        "batch_size",
        "benchmark_engine",
        "concurrency",
        "exec_env",
        "input",
        "processors",
        "requests",
        "workers",
    ]

    optional_config_keys = ["url", "dockerhub_image", "docker_dev_image", "compile_per_batch_size"]

    valid_config_keys = mandatory_config_keys + optional_config_keys

    mutually_exclusive_docker_config_keys = ["dockerhub_image", "docker_dev_image"]

    valid_batch_sizes = [1, 2, 3, 4]

    valid_processors = ["cpu", "gpus"]

    valid_docker_processors = ["cpu", "gpu", "inferentia"]

    mandatory_docker_config_keys = ["docker_tag"]

    optional_docker_config_keys = ["cuda_version"]

    valid_docker_config_keys = mandatory_docker_config_keys + optional_docker_config_keys

    @staticmethod
    def load_yaml(file_path):
        """
        :param file_path: file to load in the yaml file
        :return yaml_dict: dictonary with contents of yaml file
        """
        with open(file_path) as f:
            yaml_dict = yaml.safe_load(f)
        return yaml_dict

    @staticmethod
    def write_yaml(file_path, dictionary_object):
        """
        :param file_path: the path of the output yaml file
        :param dictionary_object: dictionary with content that needs to be written to a yaml file
        :return None
        """
        with open(file_path, "a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            yaml.dump(dictionary_object, f)
            fcntl.flock(f, fcntl.LOCK_UN)

    @staticmethod
    def validate_benchmark_yaml(yaml_content):
        """
        :param yaml_content: dictionary of benchmark config read from yaml file
        :return boolean: True/False
        """

        mode_list = []
        config_list = []
        batch_size_list = []
        processor_list = []

        for model, config in yaml_content.items():
            for mode, mode_config in config.items():
                mode_list.append(mode)
                for config, value in mode_config.items():
                    if config == "batch_size":
                        batch_size_list.append(value)
                    if config == "processors":
                        processor_list.append(value)
                    config_list.append(config)

                assert (
                    type(batch_size_list[0]) == list
                ), f"Batch sizes should be part of a list. Check config for '{mode}' under model '{model}'."

                invalid_config_keys = set(config_list).difference(YamlHandler.valid_config_keys)
                assert (
                    len(invalid_config_keys) == 0
                ), f"Invalid key(s) detected: {invalid_config_keys}. Config keys must be either of {YamlHandler.valid_config_keys}. Check config for '{mode}' under model '{model}'."

                missing_config_keys = set(YamlHandler.mandatory_config_keys).difference(config_list)
                assert (
                    len(missing_config_keys) == 0
                ), f"Config key(s) missing: {missing_config_keys}. All of the following keys are required: {YamlHandler.mandatory_config_keys}. Check config for '{mode}' under model '{model}'."

                assert not all(
                    key in config_list for key in YamlHandler.mutually_exclusive_docker_config_keys
                ), f"Either of the keys {YamlHandler.mutually_exclusive_docker_config_keys} maybe present in config, but not both. Check config for '{mode}' under model '{model}'."

                config_list.clear()
                batch_size_list.clear()
                processor_list.clear()

        invalid_mode_keys = set(mode_list).difference(YamlHandler.valid_mode_keys)
        assert (
            len(invalid_mode_keys) == 0
        ), f"Invalid mode key found, 'mode' must be either of {YamlHandler.valid_mode_keys}. Check config for '{invalid_mode_keys}' under model '{model}'"

        LOGGER.info(f"Benchmark YAML config validation successful.")

        return True

    @staticmethod
    def validate_docker_yaml(yaml_content):
        """
        :param yaml_content: dictionary containing yaml contents of the docker config
        """
        LOGGER.info(f"docker content: ")
        LOGGER.info(f"{yaml_content}")

        processor_list = []
        docker_config_list = []

        for processor, docker_config in yaml_content.items():
            processor_list.append(processor)
            for config_key, config_value in docker_config.items():
                docker_config_list.append(config_key)

            invalid_config_keys = set(docker_config_list).difference(YamlHandler.valid_docker_config_keys)
            assert (
                len(invalid_config_keys) == 0
            ), f"Invalid config key(s) detected: {invalid_config_keys}. Config keys must be either of {YamlHandler.valid_docker_config_keys}. Check config for '{processor}''."

            missing_config_keys = set(YamlHandler.mandatory_docker_config_keys).difference(docker_config_list)
            assert (
                len(missing_config_keys) == 0
            ), f"Config key(s) missing: {missing_config_keys}. All of the following keys are required: {YamlHandler.mandatory_docker_config_keys}. Check config for '{processor}'."

            if processor == "gpu":
                assert (
                    "cuda_version" in docker_config_list
                ), f"cuda_version missing under processor 'gpu'. cuda_version must be of format cuXYZ e.g.cu102, cu111 etc."

        invalid_processor_keys = set(processor_list).difference(YamlHandler.valid_docker_processors)
        assert (
            len(invalid_processor_keys) == 0
        ), f"Invalid processor key found, must be either of {YamlHandler.valid_docker_processors}"

        LOGGER.info(f"Docker YAML config validation successful.")
