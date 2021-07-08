import argparse
import os
import random
import sys
import logging
import re
import uuid


import boto3
import pytest

from invoke import run
from invoke.context import Context


from tests.utils.report import Report
from tests.utils import (
    S3_BUCKET_BENCHMARK_ARTIFACTS,
    DEFAULT_REGION,
    DEFAULT_DOCKER_DEV_ECR_REPO,
    YamlHandler,
    DockerImageHandler,
)

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)
LOGGER.addHandler(logging.StreamHandler(sys.stdout))


def build_docker_container(torchserve_branch="master"):
    LOGGER.info(f"Setting up docker image to be used")

    docker_dev_image_config_path = os.path.join(os.getcwd(), "test", "benchmark", "tests", "suite", "docker", "docker.yaml")

    docker_config = YamlHandler.load_yaml(docker_dev_image_config_path)
    YamlHandler.validate_docker_yaml(docker_config)

    account_id = run("aws sts get-caller-identity --query Account --output text").stdout.strip()

    for processor, config in docker_config.items():
        docker_tag = None
        cuda_version = None
        for config_key, config_value in config.items():
            if processor == "gpu" and config_key == "cuda_version":
                cuda_version = config_value
            if config_key == "docker_tag":
                docker_tag = config_value
        dockerImageHandler = DockerImageHandler(docker_tag, cuda_version, torchserve_branch)
        dockerImageHandler.build_image()
        dockerImageHandler.push_docker_image_to_ecr(
            account_id, DEFAULT_REGION, f"{DEFAULT_DOCKER_DEV_ECR_REPO}:{docker_tag}"
        )


def main():

    LOGGER.info(f"sys.path: {sys.path}")

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--use-instances",
        action="store",
        help="Supply a .yaml file with test_name, instance_id, and key_filename to re-use already-running instances",
    )
    parser.add_argument(
        "--do-not-terminate",
        action="store_true",
        default=False,
        help="Use with caution: does not terminate instances, instead saves the list to a file in order to re-use",
    )

    parser.add_argument(
        "--run-only", default=None, help="Runs the tests that contain the supplied keyword as a substring"
    )

    parser.add_argument(
        "--use-torchserve-branch",
        default="master",
        help="Specify a specific torchserve branch to benchmark on, else uses 'master' by default"
    )

    parser.add_argument(
        "--skip-docker-build",
        action="store_true",
        default=False,
        help="Use if you already have a docker image built and available locally and have specified it in docker.yaml"
    )

    arguments = parser.parse_args()
    do_not_terminate_string = "" if not arguments.do_not_terminate else "--do-not-terminate"
    use_instances_arg_list = ["--use-instances", f"{arguments.use_instances}"] if arguments.use_instances else []
    run_only_test = arguments.run_only

    if run_only_test:
        run_only_string = f"-k {run_only_test}"
        LOGGER.info(f"Note: running only the tests that have the name '{run_only_test}'.")
    else:
        run_only_string = ""

    torchserve_branch = arguments.use_torchserve_branch

    # Build docker containers as specified in docker.yaml
    if not arguments.skip_docker_build:
        build_docker_container(torchserve_branch=torchserve_branch)

    # Run this script from the root directory 'serve', it changes directory below as required
    os.chdir(os.path.join(os.getcwd(), "test", "benchmark"))

    execution_id = f"ts-benchmark-run-{str(uuid.uuid4())}"

    test_path = os.path.join(os.getcwd(), "tests")
    LOGGER.info(f"Running tests from directory: {test_path}")

    pytest_args = [
        "-s",
        run_only_string,
        "-rA",
        test_path,
        "-n=4",
        "--disable-warnings",
        "-v",
        "--execution-id",
        execution_id,
        do_not_terminate_string,
    ] + use_instances_arg_list

    LOGGER.info(f"Running pytest")

    pytest.main(pytest_args)

    # Generate report
    s3_results_uri = f"{S3_BUCKET_BENCHMARK_ARTIFACTS}/{execution_id}"

    report = Report()
    report.download_benchmark_results_from_s3(s3_results_uri)
    report.generate_comprehensive_report()


if __name__ == "__main__":
    main()
