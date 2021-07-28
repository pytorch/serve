import subprocess
import time
import glob
import os
import requests
import tempfile

import invoke
import pandas as pd

from io import StringIO
from urllib.parse import urlparse
from invoke import run
from invoke.context import Context

from . import DEFAULT_REGION, IAM_INSTANCE_PROFILE, AMI_ID, LOGGER, S3_BUCKET_BENCHMARK_ARTIFACTS

# Assumes the functions from this file execute on an Ubuntu ec2 instance
ROOT_DIR = f"/home/ubuntu"
TORCHSERVE_DIR = os.path.join(ROOT_DIR, "serve")
MODEL_STORE = os.path.join(TORCHSERVE_DIR, "model_store")
LOCAL_TMP_DIR = "/tmp"
TMP_DIR = "/home/ubuntu"


class TorchServeHandler(object):
    def __init__(
        self,
        exec_env="docker",
        cuda_version="cu102",
        gpus=None,
        torchserve_docker_image=None,
        backend_profiling=None,
        connection=None,
    ):
        self.exec_env = exec_env
        self.cuda_version = cuda_version
        self.gpus = gpus
        self.torchserve_docker_image = torchserve_docker_image
        self.backend_profiling = backend_profiling
        self.connection = invoke if not connection else connection
        self.config_properties = os.path.join(TMP_DIR, "benchmark", "conf", "config.properties")

        self.management_api = "http://127.0.0.1:8081"
        self.inference_api = "http://127.0.0.1:8080"
        self.management_port = urlparse(self.management_api).port
        self.inference_port = urlparse(self.inference_api).port

        # Following sequence of calls are important
        # self.prepare_common_dependency()
        # self.getAPIS()

    def setup_torchserve(self, virtual_env_name=None):
        """
        Set up torchserve dependencies, and install torchserve
        """
        activation_command = ""
        self.connection.run(f"chmod +x -R /home/ubuntu/serve")
        if virtual_env_name:
            activation_command = f"cd /home/ubuntu/serve && source activate {virtual_env_name} && "

        if self.connection.run(f"{activation_command}torchserve --version", warn=True).return_code == 0:
            return

        self.connection.run(f"{activation_command}python3 ./ts_scripts/install_dependencies.py --environment=dev", warn=True)
        self.connection.run(f"{activation_command}pip3 install pygit2", warn=True)
        self.connection.run(f"{activation_command}python3 ./ts_scripts/install_from_src.py", warn=True)
        self.connection.run(f"{activation_command}torchserve --version")


    def prepare_common_dependency(self):
        # Note: the following command cleans up any previous run logs, except any *.mar files generated to avoid re-creation
        self.connection.run(f"find {os.path.join(TMP_DIR, 'benchmark')} ! -name '*.mar' -type f -exec rm -f {{}} +", warn=True)
        # Recreate required folders
        self.connection.run(f"mkdir -p {os.path.join(TMP_DIR, 'benchmark', 'conf')}")
        self.connection.run(f"mkdir -p {os.path.join(TMP_DIR, 'benchmark', 'logs')}")
        self.connection.run(f"mkdir -p {os.path.join(TMP_DIR, 'benchmark', 'model_store')}")

        # Use config from benchmarks/ folder
        self.connection.run(
            f"cp {os.path.join(TORCHSERVE_DIR, 'benchmarks', 'config.properties')} {os.path.join(TMP_DIR, 'benchmark', 'conf')}"
        )

    def getAPIS(self):
        self.connection.get(self.config_properties, os.path.join(LOCAL_TMP_DIR, "config.properties"))

        with open(os.path.join(LOCAL_TMP_DIR, "config.properties")) as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip()
            if "management_address" in line:
                management_api = line.split("=")[1]
            if "inference_address" in line:
                inference_api = line.split("=")[1]

        self.management_api = management_api
        self.inference_api = inference_api

        self.management_port = urlparse(management_api).port
        self.inference_api = urlparse(inference_api).port

    def start_torchserve_local(self, virtual_env_name=None, stop_torchserve=True):

        self.prepare_common_dependency()
        self.getAPIS()

        activation_command = ""
        if virtual_env_name:
            activation_command = f"cd /home/ubuntu/serve && source activate {virtual_env_name} && "
        if self.backend_profiling:
            activation_command = f"{activation_command} && export TS_BENCHMARK=True && "
        
        if stop_torchserve:
            LOGGER.info(f"Stop existing torchserve instance")
            self.connection.run(f"{activation_command}torchserve --stop", warn=True)
        
        self.connection.run(f"{activation_command}torchserve --start --model-store /home/ubuntu/benchmark/model_store/ --ts-config {TMP_DIR}/benchmark/conf/config.properties > {TMP_DIR}/benchmark/logs/model_metrics.log", warn=True)
        LOGGER.info(f"Started torchserve using command")
        LOGGER.info(f"{activation_command}torchserve --start --model-store /home/ubuntu/benchmark/model_store/ --ts-config {TMP_DIR}/benchmark/conf/config.properties > {TMP_DIR}/benchmark/logs/model_metrics.log")

        time.sleep(10)


    def start_torchserve_docker(self, stop_torchserve=True):

        self.prepare_common_dependency()
        self.getAPIS()

        enable_gpu = ""
        backend_profiling = ""
        if self.cuda_version and self.gpus:
            enable_gpu = f"--gpus {self.gpus}"
        if self.backend_profiling:
            backend_profiling = f"-e TS_BENCHMARK=True"

        if stop_torchserve:
            LOGGER.info(f"Removing existing TS container instance...")
            self.connection.run("docker rm -f ts")

        LOGGER.info(f"Starting docker container on the instance from image: {self.torchserve_docker_image}")
        docker_run_cmd = (
            f"docker run {backend_profiling} --name ts --user root -p {self.inference_port}:{self.inference_port} -p {self.management_port}:{self.management_port} "
            f"-v {TMP_DIR}:/tmp {enable_gpu} -itd {self.torchserve_docker_image} "
            f'"torchserve --start --model-store /home/model-server/model-store '
            f'--ts-config /tmp/benchmark/conf/config.properties > /tmp/benchmark/logs/model_metrics.log"'
        )

        LOGGER.info(f"Logging docker run command: {docker_run_cmd}")

        self.connection.run(docker_run_cmd)

        time.sleep(8)

    def register_model(self, url, workers, batch_delay, batch_size, model_name="benchmark"):
        """
        Uses 'curl' on the connection to register model
        :param url: http url for the pre-trained model
        :param workers: number of torchserve workers to use
        :param batch_delay: batch_delay allowed for requests
        :param batch_size: max number of requests allowed to be batched
        """
        run_out = self.connection.run(
            f'curl -X POST "http://localhost:8081/models?url={url}&initial_workers={workers}&batch_delay={batch_delay}&batch_size={batch_size}&synchronous=true&model_name=benchmark"', warn=True
        )

        LOGGER.info(
            f'curl -X POST "http://localhost:8081/models?url={url}&initial_workers={workers}&batch_delay={batch_delay}&batch_size={batch_size}&synchronous=true&model_name=benchmark"'
        )

        time.sleep(40)

        if run_out.return_code == 0:
            LOGGER.error(f"Failed to register model {model_name} sourced from url: {url}")

    def unregister_model(self, model_name="benchmark"):
        """
        Uses 'curl' on the connection to unregister the model. Assumes only a single version of the model is loaded.
        Typically should be run after every benchmark configuration completes. 
        :param model_name: The name of the model to unregister
        """
        run_out = self.connection.run(f'curl -X DELETE "http://localhost:8081/models/{model_name}/1.0"', warn=True)
        LOGGER.info(f'curl -X DELETE "http://localhost:8081/models/{model_name}/1.0"')
        LOGGER.info(f"stdout: {run_out.stdout}")

        time.sleep(10)
        if run_out.return_code == 0:
            LOGGER.error(f"Failed to unregister model {model_name}")


    def stop_torchserve(self, exec_env="docker", virtual_env_name=None):
        """
        Stops torchserve depending on the exec_env
        :param exec_env: either 'local' or 'docker'
        """
        if exec_env == "docker":
            self.connection.run(f"docker rm -f ts", warn=True)
        else:
            activation_command = ""
            if virtual_env_name:
                activation_command = f"cd /home/ubuntu/serve/test/benchmark/tests/resources/neuron-bert && source activate {virtual_env_name} && "
            self.connection.run(f"{activation_command}torchserve --stop", warn=True)

        time.sleep(5)


def delete_mar_file_from_model_store(model_store=None, model_mar=None):
    model_store = model_store if (model_store is not None) else f"{ROOT_DIR}/model_store/"
    if model_mar is not None:
        for f in glob.glob(os.path.join(model_store, model_mar + "*")):
            os.remove(f)
