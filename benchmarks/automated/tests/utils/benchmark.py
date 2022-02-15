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
    def __init__(self, model_name, benchmark_execution_id, connection=None, is_local_execution=False, benchmark_type="docker"):
        """
        :param model_name: Name of the model to be benchmarked
        :param benchmark_execution_id: execution id that is shared across all the tests running 
                                        in the current suite
        :param connection: fabric/invoke connection for local or remote execution
        :param is_local_execution: boolean that specifies if the benchmark suite is already running on the
                                    instance against which it should benchmark
        :param benchmark_type: this specifies if torchserve should be setup and used in 'virtual_env', supplied 'docker'
                                image, or if a 'workflow' is being benchmarked. Note: 
        """
        self.model_name = model_name
        self.benchmark_execution_id = benchmark_execution_id
        self.connection = invoke if not connection else connection
        self.is_local_execution = is_local_execution

        self.is_workflow = False
        self.is_docker_execution = False
        self.is_virtual_env_execution = False

    def execute_benchmark(
        self,
        test_config,
        ec2_instance_type,
        cuda_version_for_instance,
        docker_repo_tag_for_current_instance,
        exec_env="docker",
        local_virutal_env="python3",
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

        if self.is_local_execution:
            ec2_instance_type = "local_execution"
            LOGGER.info(f"*** Note: Executing benchmark on the current instance.")
        else:
            ec2_instance_type = ec2_instance_type

        ec2_instance_type = "local_execution" if self.is_local_execution else ec2_instance_type
        apacheBenchHandler = ab_utils.ApacheBenchHandler(model_name=self.model_name, connection=self.connection)

        for model, config in test_config.items():
            if model == "instance_types":
                continue
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
                    f"\n benchmark_engine: {benchmark_engine}\n  workers: {workers}\n batch_delay: {batch_delay}\n batch_size:{batch_sizes}\n input_file: {input_file}\n requests: {requests}\n concurrency: {concurrency}\n backend_profiling: {backend_profiling}\n exec_env: {exec_env}\n processors: {processors}"
                )

                # Assign type of benchmark
                if exec_env == "docker":
                    self.is_docker_execution = True
                else:
                    self.is_virtual_env_execution = True
                    local_virutal_env = exec_env

                torchserveHandler = ts_utils.TorchServeHandler(
                    exec_env=exec_env,
                    cuda_version=cuda_version_for_instance,
                    gpus=gpus,
                    torchserve_docker_image=docker_repo_tag_for_current_instance,
                    backend_profiling=backend_profiling,
                    connection=self.connection,
                    is_local_execution=self.is_local_execution,
                )

                # Note: Assumes a DLAMI (conda-based) is being used
                if exec_env != "docker":
                    torchserveHandler.setup_torchserve(virtual_env_name=exec_env)

                for batch_size in batch_sizes:
                    if "inferentia" in processors:
                        url = f"benchmark_{batch_size}.mar"
                        LOGGER.info(f"Running benchmark for model archive: {url}")

                    # Stop torchserve
                    torchserveHandler.stop_torchserve(exec_env=exec_env, virtual_env_name=exec_env)

                    # Generate bert inf model
                    if "neuron" in exec_env:
                        neuron_utils.setup_neuron_mar_files(
                            connection=self.connection, virtual_env_name=exec_env, batch_size=batch_size
                        )

                    # Start torchserve
                    if exec_env != "docker":
                        torchserveHandler.start_torchserve_local(
                            virtual_env_name=exec_env, stop_torchserve=False
                        )
                    else:
                        torchserveHandler.start_torchserve_docker()
                        torchserveHandler.start_recording_docker_stats()

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
                    if exec_env == "docker":
                        torchserveHandler.stop_recording_docker_stats(model_name=self.model_name, num_workers=workers, batch_size=batch_size)
                        torchserveHandler.stop_torchserve()
                        torchserveHandler.plot_stats_graph(model_name=self.model_name, mode_name = mode,num_workers=workers, batch_size=batch_size)
                    else:
                        torchserveHandler.stop_torchserve(exec_env="local", virtual_env_name=exec_env)

                    # Generate report (note: needs to happen after torchserve has stopped)
                    apacheBenchHandler.generate_report(
                        requests=requests, concurrency=concurrency, batch_size=batch_size, mode=mode,connection=self.connection
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
