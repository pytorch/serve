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

# Add/remove from the following list to benchmark on the instance of your choice
INSTANCE_TYPES_TO_TEST = ["p3.8xlarge"]


@pytest.mark.parametrize("ec2_instance_type", INSTANCE_TYPES_TO_TEST, indirect=True)
def test_bert_benchmark(
    ec2_connection, ec2_instance_type, bert_config_file_path, docker_dev_image_config_path, benchmark_execution_id
):

    test_config = YamlHandler.load_yaml(bert_config_file_path)

    model_name = bert_config_file_path.split("/")[-1].split(".")[0]

    LOGGER.info("Validating yaml contents")

    LOGGER.info(YamlHandler.validate_model_yaml(test_config))

    docker_config = YamlHandler.load_yaml(docker_dev_image_config_path)

    cuda_version_for_instance, docker_repo_tag_for_current_instance = DockerImageHandler.process_docker_config(
        ec2_connection, docker_dev_image_config_path, ec2_instance_type
    )

    benchmarkHandler = benchmark_utils.BenchmarkHandler(model_name, benchmark_execution_id, ec2_connection)

    benchmarkHandler.execute_docker_benchmark(
        test_config, ec2_instance_type, cuda_version_for_instance, docker_repo_tag_for_current_instance
    )
