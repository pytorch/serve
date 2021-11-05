import os
import pprint

import pytest
import time
from invoke import run
from invoke.context import Context

import tests.utils.benchmark as benchmark_utils

from tests.utils import (
    DEFAULT_DOCKER_DEV_ECR_REPO,
    DEFAULT_REGION,
    GPU_INSTANCES,
    LOGGER,
    DockerImageHandler,
    YamlHandler,
    S3_BUCKET_BENCHMARK_ARTIFACTS,
)


def test_model_benchmark(
    ec2_connection, model_config_path_ec2_instance_tuple, docker_dev_image_config_path, benchmark_execution_id, is_local_execution
):
    (model_config_file_path, ec2_instance_type) = model_config_path_ec2_instance_tuple

    test_config = YamlHandler.load_yaml(model_config_file_path)

    model_name = model_config_file_path.split("/")[-1].split(".")[0]

    LOGGER.info("Validating yaml contents")

    LOGGER.info(YamlHandler.validate_model_yaml(test_config))

    cuda_version_for_instance, docker_repo_tag_for_current_instance = DockerImageHandler.process_docker_config(
        ec2_connection, docker_dev_image_config_path, ec2_instance_type, is_local_execution
    )

    benchmarkHandler = benchmark_utils.BenchmarkHandler(model_name, benchmark_execution_id, ec2_connection, is_local_execution)

    benchmarkHandler.execute_benchmark(
        test_config, ec2_instance_type, cuda_version_for_instance, docker_repo_tag_for_current_instance
    )
