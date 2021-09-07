import subprocess
import time
import glob
import os
import re
import requests
import tempfile
import uuid

import invoke
import pandas as pd

from io import StringIO
from pathlib import Path
from urllib.parse import urlparse
from invoke import run
from invoke.context import Context

from . import DEFAULT_REGION, IAM_INSTANCE_PROFILE, AMI_ID, LOGGER, S3_BUCKET_BENCHMARK_ARTIFACTS

from . import apache_bench as ab_utils
from . import ts as ts_utils
from . import neuron as neuron_utils


class BenchmarkHandler:
    def __init__(self, model_name, benchmark_execution_id, connection=None):
        """
        :param connection: fabric/invoke connection for local or remote execution
        """
        self.model_name = model_name
        self.benchmark_execution_id = benchmark_execution_id
        self.connection = invoke if not connection else connection

    def execute_local_benchmark(
        self,
        test_config,
        ec2_instance_type,
        cuda_version_for_instance,
        docker_repo_tag_for_current_instance,
        exec_env="python3",
    ):
        """
        :param test_config: config for the specific model to be benchmarked
        :param ec2_instance_type: instance type to be run on, determines whether cpu or gpu dockerfile is used
        :param cuda_version_for_instance: cuda_version, if any
        :param docker_repo_tag_for_current_instance: ecr repo tag to pull the container image from
        :param exec_env: name of the already-existing virtual environment to use on a DLAMI
        """
        mode_list = []
        config_list = []
        batch_size_list = []
        processor_list = []

        apacheBenchHandler = ab_utils.ApacheBenchHandler(model_name=self.model_name, connection=self.connection)

        for model, config in test_config.items():
            if model == "instance_types":
                continue
            for mode, mode_config in config.items():
                mode_list.append(mode)
                benchmark_engine = mode_config.get("benchmark_engine")
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
                    f"\n benchmark_engine: {benchmark_engine}\n  workers: {workers}\n batch_delay: {batch_delay}\n batch_size:{batch_sizes}\n input_file: {input_file}\n requests: {requests}\n concurrency: {concurrency}\n backend_profiling: {backend_profiling}\n exec_env: {exec_env}\n processors: {processors}"
                )

                torchserveHandler = ts_utils.TorchServeHandler(
                    exec_env=exec_env,
                    cuda_version=cuda_version_for_instance,
                    gpus=gpus,
                    torchserve_docker_image=docker_repo_tag_for_current_instance,
                    backend_profiling=backend_profiling,
                    connection=self.connection,
                )

                # Note: Assumes a DLAMI (conda-based) is being used
                torchserveHandler.setup_torchserve(virtual_env_name="aws_neuron_pytorch_p36")

                for batch_size in batch_sizes:
                    url = f"benchmark_{batch_size}.mar"
                    LOGGER.info(f"Running benchmark for model archive: {url}")

                    # Stop torchserve
                    torchserveHandler.stop_torchserve(exec_env="local", virtual_env_name="aws_neuron_pytorch_p36")

                    # Generate bert inf model
                    neuron_utils.setup_neuron_mar_files(
                        connection=self.connection, virtual_env_name="aws_neuron_pytorch_p36", batch_size=batch_size
                    )

                    # Start torchserve
                    torchserveHandler.start_torchserve_local(
                        virtual_env_name="aws_neuron_pytorch_p36", stop_torchserve=False
                    )

                    # Register
                    torchserveHandler.register_model(
                        url=url, workers=workers, batch_delay=batch_delay, batch_size=batch_size
                    )

                    # Run benchmark
                    apacheBenchHandler.run_apache_bench(
                        requests=requests, concurrency=concurrency, input_file=input_file
                    )

                    # Unregister
                    torchserveHandler.unregister_model()

                    # Stop torchserve
                    torchserveHandler.stop_torchserve(exec_env="local", virtual_env_name="aws_neuron_pytorch_p36")

                    # Generate report (note: needs to happen after torchserve has stopped)
                    apacheBenchHandler.generate_report(
                        requests=requests, concurrency=concurrency, connection=self.connection
                    )

                    # Move artifacts into a common folder.
                    remote_artifact_folder = f"/home/ubuntu/{self.benchmark_execution_id}/{self.model_name}/{ec2_instance_type}/{mode}/{batch_size}"

                    self.connection.run(f"mkdir -p {remote_artifact_folder}")
                    self.connection.run(f"cp -R /home/ubuntu/benchmark/* {remote_artifact_folder}")

                    # Upload artifacts to s3 bucket
                    self.connection.run(
                        f"aws s3 cp --recursive /home/ubuntu/{self.benchmark_execution_id}/ {S3_BUCKET_BENCHMARK_ARTIFACTS}/{self.benchmark_execution_id}/"
                    )

                    time.sleep(3)

                    run(
                        f"aws s3 cp --recursive /tmp/{self.model_name}/ {S3_BUCKET_BENCHMARK_ARTIFACTS}/{self.model_name}/{ec2_instance_type}/{mode}/{batch_size}"
                    )

                    run(f"rm -rf /tmp/{self.model_name}")
                    apacheBenchHandler.clean_up()

    def execute_docker_benchmark(
        self, test_config, ec2_instance_type, cuda_version_for_instance, docker_repo_tag_for_current_instance
    ):
        """
        :param test_config: config for the specific model to be benchmarked
        :param ec2_instance_type: instance type to be run on, determines whether cpu or gpu dockerfile is used
        :param cuda_version_for_instance: cuda version, if any
        :param docker_repo_tag_for_current_instance: ecr repo tag to pull the container image from
        """
        mode_list = []
        config_list = []
        batch_size_list = []
        processor_list = []

        apacheBenchHandler = ab_utils.ApacheBenchHandler(model_name=self.model_name, connection=self.connection)

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
                    connection=self.connection,
                )

                for batch_size in batch_sizes:

                    # Start torchserve
                    torchserveHandler.start_torchserve_docker()

                    # Register
                    torchserveHandler.register_model(
                        url=url, workers=workers, batch_delay=batch_delay, batch_size=batch_size
                    )

                    # Run benchmark
                    apacheBenchHandler.run_apache_bench(
                        requests=requests, concurrency=concurrency, input_file=input_file
                    )

                    # Unregister
                    torchserveHandler.unregister_model()

                    # Stop torchserve
                    torchserveHandler.stop_torchserve()

                    # Generate report (note: needs to happen after torchserve has stopped)
                    apacheBenchHandler.generate_report(
                        requests=requests, concurrency=concurrency, connection=self.connection
                    )

                    # Move artifacts into a common folder.
                    remote_artifact_folder = f"/home/ubuntu/{self.benchmark_execution_id}/{self.model_name}/{ec2_instance_type}/{mode}/{batch_size}"

                    self.connection.run(f"mkdir -p {remote_artifact_folder}")
                    self.connection.run(f"cp -R /home/ubuntu/benchmark/* {remote_artifact_folder}")

                    # Upload artifacts to s3 bucket
                    self.connection.run(
                        f"aws s3 cp --recursive /home/ubuntu/{self.benchmark_execution_id}/ {S3_BUCKET_BENCHMARK_ARTIFACTS}/{self.benchmark_execution_id}/"
                    )

                    time.sleep(3)

                    run(
                        f"aws s3 cp --recursive /tmp/{self.model_name}/ {S3_BUCKET_BENCHMARK_ARTIFACTS}/{self.benchmark_execution_id}/{self.model_name}/{ec2_instance_type}/{mode}/{batch_size}"
                    )

                    run(f"rm -rf /tmp/{self.model_name}")
                    apacheBenchHandler.clean_up()


    def execute_workflow_benchmark(
        self, test_config, ec2_instance_type, cuda_version_for_instance, docker_repo_tag_for_current_instance
    ):
        """
        :param test_config: config for the specific model to be benchmarked
        :param ec2_instance_type: instance type to be run on, determines whether cpu or gpu dockerfile is used
        :param cuda_version_for_instance: cuda version, if any
        :param docker_repo_tag_for_current_instance: ecr repo tag to pull the container image from
        """
        mode_list = []
        config_list = []
        batch_size_list = []
        processor_list = []

        apacheBenchHandler = ab_utils.ApacheBenchHandler(model_name=self.model_name, connection=self.connection)

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

                # values for workflow
                workflow_name = mode_config.get("workflow_name")
                workflow_model_urls = mode_config.get("models")
                workflow_specfile_url = mode_config.get("specfile")
                workflow_handler_url = mode_config.get("workflow_handler")
                retry_attempts = mode_config.get("retry_attempts")
                timeout_ms = mode_config.get("timeout_ms")

                # url is just a name in this case
                url = workflow_name

                LOGGER.info(f"model_urls_in_workflow: {workflow_model_urls}")
                LOGGER.info(f"workflow_specfile_url: {workflow_specfile_url}")
                LOGGER.info(f"workflow_handler_url: {workflow_handler_url}")
                LOGGER.info(f"workflow_name: {workflow_name}")

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
                    connection=self.connection,
                )

                torchserveHandler.download_workflow_artifacts(
                    workflow_name, workflow_model_urls, workflow_specfile_url, workflow_handler_url
                )

                for batch_size in batch_sizes:
                    # Start torchserve
                    torchserveHandler.start_torchserve_docker()

                    # Create workflow archive and place in the wf_store
                    torchserveHandler.create_and_update_workflow_archive(
                        workflow_name,
                        os.path.basename(workflow_specfile_url),
                        os.path.basename(workflow_handler_url),
                        batch_size,
                        workers,
                        batch_delay,
                        retry_attempts,
                        timeout_ms,
                    )

                    # Register
                    torchserveHandler.register_workflow(url=url)

                    # Run benchmark
                    apacheBenchHandler.run_apache_bench(
                        requests=requests,
                        concurrency=concurrency,
                        input_file=input_file,
                        is_workflow=True,
                        workflow_name=workflow_name,
                    )

                    # Unregister
                    torchserveHandler.unregister_workflow(workflow_name=workflow_name)

                    # Stop torchserve
                    torchserveHandler.stop_torchserve()

                    # Generate report (note: needs to happen after torchserve has stopped)
                    apacheBenchHandler.generate_report(
                        requests=requests, concurrency=concurrency, connection=self.connection
                    )

                    # Move artifacts into a common folder.
                    remote_artifact_folder = f"/home/ubuntu/{self.benchmark_execution_id}/{self.model_name}/{ec2_instance_type}/{mode}/{batch_size}"

                    self.connection.run(f"mkdir -p {remote_artifact_folder}")
                    self.connection.run(f"cp -R /home/ubuntu/benchmark/* {remote_artifact_folder}")

                    # Upload artifacts to s3 bucket
                    self.connection.run(
                        f"aws s3 cp --recursive /home/ubuntu/{self.benchmark_execution_id}/ {S3_BUCKET_BENCHMARK_ARTIFACTS}/{self.benchmark_execution_id}/"
                    )

                    time.sleep(3)

                    run(
                        f"aws s3 cp --recursive /tmp/{self.model_name}/ {S3_BUCKET_BENCHMARK_ARTIFACTS}/{self.benchmark_execution_id}/{self.model_name}/{ec2_instance_type}/{mode}/{batch_size}"
                    )

                    run(f"rm -rf /tmp/{self.model_name}")
                    apacheBenchHandler.clean_up()
