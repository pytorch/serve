import os
import pprint

import pytest
import time
from invoke import run
from invoke.context import Context

import tests.utils.ec2 as ec2_utils
import tests.utils.s3 as s3_utils
import tests.utils.ts as ts_utils
import tests.utils.apache_bench as ab_utils

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
INSTANCE_TYPES_TO_TEST = ["c4.4xlarge"]

@pytest.mark.parametrize("ec2_instance_type", INSTANCE_TYPES_TO_TEST, indirect=True)
def test_vgg16_benchmark(
    ec2_connection, ec2_instance_type, vgg16_config_file_path, docker_dev_image_config_path, benchmark_execution_id
):

    test_config = YamlHandler.load_yaml(vgg16_config_file_path)

    model_name = vgg16_config_file_path.split("/")[-1].split(".")[0]

    LOGGER.info("Validating yaml contents")

    LOGGER.info(YamlHandler.validate_benchmark_yaml(test_config))

    docker_config = YamlHandler.load_yaml(docker_dev_image_config_path)

    docker_repo_tag_for_current_instance = ""
    cuda_version_for_instance = ""
    account_id = run("aws sts get-caller-identity --query Account --output text").stdout.strip()

    for processor, config in docker_config.items():
        docker_tag = None
        cuda_version = None
        for config_key, config_value in config.items():
            if processor == "gpu" and config_key == "cuda_version":
                cuda_version = config_value
            if config_key == "docker_tag":
                docker_tag = config_value
        # TODO: Improve logic that selectively pulls CPU image on CPU instances and likewise for GPU.

        docker_repo_tag = f"{DEFAULT_DOCKER_DEV_ECR_REPO}:{docker_tag}"

        if ec2_instance_type[:2] in GPU_INSTANCES and ("gpu" in docker_tag or "neuron" in docker_tag):
            dockerImageHandler = DockerImageHandler(docker_tag, cuda_version)
            dockerImageHandler.pull_docker_image_from_ecr(
                account_id, DEFAULT_REGION, docker_repo_tag, connection=ec2_connection
            )
            docker_repo_tag_for_current_instance = docker_repo_tag
            cuda_version_for_instance = cuda_version
            break
        if ec2_instance_type[:2] not in GPU_INSTANCES and ("cpu" in docker_tag or "neuron" in docker_tag):
            dockerImageHandler = DockerImageHandler(docker_tag, cuda_version)
            dockerImageHandler.pull_docker_image_from_ecr(
                account_id, DEFAULT_REGION, docker_repo_tag, connection=ec2_connection
            )
            docker_repo_tag_for_current_instance = docker_repo_tag
            cuda_version_for_instance = None
            break

    mode_list = []
    config_list = []
    batch_size_list = []
    processor_list = []

    apacheBenchHandler = ab_utils.ApacheBenchHandler(model_name=model_name, connection=ec2_connection)

    for model, config in test_config.items():
        for mode, mode_config in config.items():
            mode_list.append(mode)
            benchmark_engine = mode_config.get("benchmark_engine")
            url = mode_config.get("url")
            workers = mode_config.get("workers")
            batch_delay = mode_config.get("batch_delay")
            batch_sizes = mode_config.get("batch_size")
            input_file = mode_config.get("input")
            requests = mode_config.get("requests")
            concurrency = mode_config.get("concurrency")
            backend_profiling = mode_config.get("backend_profiling")
            exec_env = mode_config.get("exec_env")
            processors = mode_config.get("processors")
            gpus = None
            if len(processors) == 2:
                gpus = processors[1].get("gpus")
            LOGGER.info(f"processors: {processors[1]}")
            LOGGER.info(f"gpus: {gpus}")

            LOGGER.info(
                f"\n benchmark_engine: {benchmark_engine}\n url: {url}\n workers: {workers}\n batch_delay: {batch_delay}\n batch_size:{batch_sizes}\n input_file: {input_file}\n requests: {requests}\n concurrency: {concurrency}\n backend_profiling: {backend_profiling}\n exec_env: {exec_env}\n processors: {processors}"
            )

            torchserveHandler = ts_utils.TorchServeHandler(
                exec_env=exec_env,
                cuda_version=cuda_version_for_instance,
                gpus=gpus,
                torchserve_docker_image=docker_repo_tag_for_current_instance,
                backend_profiling=backend_profiling,
                connection=ec2_connection,
            )

            for batch_size in batch_sizes:

                # Start torchserve
                torchserveHandler.start_torchserve_docker()

                # Register
                torchserveHandler.register_model(
                    url=url, workers=workers, batch_delay=batch_delay, batch_size=batch_size
                )

                # Run benchmark
                apacheBenchHandler.run_apache_bench(requests=requests, concurrency=concurrency, input_file=input_file)

                # Unregister
                torchserveHandler.unregister_model()

                # Stop torchserve
                torchserveHandler.stop_torchserve(exec_env="docker")

                # Generate report (note: needs to happen after torchserve has stopped)
                apacheBenchHandler.generate_report(requests=requests, concurrency=concurrency, connection=ec2_connection)

                # Move artifacts into a common folder.
                remote_artifact_folder = (
                    f"/home/ubuntu/{benchmark_execution_id}/{model_name}/{ec2_instance_type}/{mode}/{batch_size}"
                )

                ec2_connection.run(f"mkdir -p {remote_artifact_folder}")
                ec2_connection.run(f"cp -R /home/ubuntu/benchmark/* {remote_artifact_folder}")

                # Upload artifacts to s3 bucket
                ec2_connection.run(
                    f"aws s3 cp --recursive /home/ubuntu/{benchmark_execution_id}/ {S3_BUCKET_BENCHMARK_ARTIFACTS}/{benchmark_execution_id}/"
                )

                time.sleep(3)

                run(
                    f"aws s3 cp --recursive /tmp/{model_name}/ {S3_BUCKET_BENCHMARK_ARTIFACTS}/{benchmark_execution_id}/{model_name}/{ec2_instance_type}/{mode}/{batch_size}"
                )

                run(f"rm -rf /tmp/{model_name}")
                apacheBenchHandler.clean_up()