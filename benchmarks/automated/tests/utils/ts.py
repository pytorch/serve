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

from . import LOGGER, S3_BUCKET_BENCHMARK_ARTIFACTS

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
        is_local_execution=False
    ):
        self.exec_env = exec_env
        self.cuda_version = cuda_version
        self.gpus = gpus
        self.torchserve_docker_image = torchserve_docker_image
        self.backend_profiling = backend_profiling
        self.connection = invoke if not connection else connection
        self.is_local_execution = is_local_execution

        self.config_properties = os.path.join(TMP_DIR, "benchmark", "conf", "config.properties")

        self.management_api = "http://127.0.0.1:8081"
        self.inference_api = "http://127.0.0.1:8080"
        self.management_port = urlparse(self.management_api).port
        self.inference_port = urlparse(self.inference_api).port
        self.tmp_wf_dir = os.path.join(LOCAL_TMP_DIR, "workflow")

        # Install torch-model-archiver and torch-workflow-archiver
        self.connection.run(f"pip3 install torch-model-archiver torch-workflow-archiver")

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
        self.connection.run(f"{activation_command}pip3 install torch-model-archiver torch-workflow-archiver")


    def prepare_common_dependency(self):
        # Note: the following command cleans up any previous run logs, except any *.mar files generated to avoid re-creation
        self.connection.run(f"find {os.path.join(TMP_DIR, 'benchmark')} ! -name '*.mar' -type f -exec rm -f {{}} +", warn=True)
        # Recreate required folders
        self.connection.run(f"mkdir -p {os.path.join(TMP_DIR, 'benchmark', 'conf')}")
        self.connection.run(f"mkdir -p {os.path.join(TMP_DIR, 'benchmark', 'logs')}")
        self.connection.run(f"mkdir -p {os.path.join(TMP_DIR, 'benchmark', 'model_store')}")
        self.connection.run(f"mkdir -p {os.path.join(TMP_DIR, 'benchmark', 'wf_store')}")

        # Use config from benchmarks/ folder
        self.connection.run(
            f"cp {os.path.join(TORCHSERVE_DIR, 'benchmarks', 'config.properties')} {os.path.join(TMP_DIR, 'benchmark', 'conf')}"
        )


    def getAPIS(self):
        if self.is_local_execution:
            self.connection.run(f"cp {self.config_properties} {os.path.join(LOCAL_TMP_DIR, 'config.properties')}")
        else:
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
        LOGGER.info(f"{activation_command}torchserve --start --model-store /home/ubuntu/benchmark/model_store/ --workflow-store /home/ubuntu/benchmark/wf_store --ts-config {TMP_DIR}/benchmark/conf/config.properties --ncs > {TMP_DIR}/benchmark/logs/model_metrics.log")

        time.sleep(10)

    
    def start_recording_docker_stats(self):
        """
        Records docker stats for the container 'ts' using nohup, in the file nohup.out 
        """
        LOGGER.info("Recording benchmark stats")
        self.connection.run("docker exec -d ts bash -c \"while true; do free |  grep -i mem | awk '{ tmp=(\$3)/(\$2) ; printf\\\"%0.2f\\n\\\", tmp }' >> /tmp/benchmark/logs/free.out; sleep 1 ; done \"",echo=True, warn=True, pty=False)
        self.connection.run("nohup bash -c 'while true; do docker stats ts --format '{{.CPUPerc}}' --no-stream | sed 's/\%//g' 2>&1; sleep 0.5; done >& nohup.out < nohup.out & '", pty=False)
        time.sleep(3)

    
    def stop_recording_docker_stats(self, model_name, num_workers, batch_size):
        """
        Stops and cleans up docker stats, and preps for plotting
        """
        # Gathers the result of free command
        self.connection.run(f"cp /home/ubuntu/benchmark/logs/free.out ./free.{model_name}.{num_workers}.{batch_size}", warn=True)

        # Stops the nohup process that was recording stats
        self.connection.run("ps axl|grep -e '--no-stream'| grep -v color | awk '{print $3}' | xargs kill -9", warn=True)
        self.connection.run(f"cp nohup.out nohup.{model_name}.{num_workers}.{batch_size}", warn=True)
        self.connection.run(f"rm nohup.out", warn=True)
        time.sleep(3)

    def plot_stats_graph(self, model_name, mode_name, num_workers, batch_size):
        """
        Plots the graphs for docker stats recorded (free, docker cpu utilization) etc.
        """
        import matplotlib.pyplot as plt

        LOGGER.info(f"Generating graphs")

        if not self.is_local_execution:
            self.connection.get(f"free.{model_name}.{num_workers}.{batch_size}", f"free.{model_name}.{num_workers}.{batch_size}")

        # plot graphs from the utility 'free'
        with open(f"free.{model_name}.{num_workers}.{batch_size}") as f:
            file_contents = [float(line.strip()) for line in f.readlines()]

        y_data = file_contents

        plt.plot(y_data)
        plt.xlabel(f"time")
        plt.ylabel(f"% utilized memory by torchserve")
        plt.title(f"{model_name} {mode_name} num_workers={num_workers} batch_size={batch_size}")
        plt.savefig(f"free_plot.{model_name}.{mode_name}.{num_workers}.{batch_size}.png")
        plt.clf()

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
            f'"torchserve --start --model-store /tmp/benchmark/model_store --workflow-store /tmp/benchmark/wf_store '
            f'--ts-config /tmp/benchmark/conf/config.properties --ncs > /tmp/benchmark/logs/model_metrics.log"'
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

        if run_out.return_code != 0:
            LOGGER.error(f"Failed to register model {model_name} sourced from url: {url}")

    def unregister_model(self, model_name="benchmark"):
        """
        Uses 'curl' on the connection to unregister the model. Assumes only a single version of the model is loaded.
        Typically should be run after every benchmark configuration completes. 
        :param model_name: The name of the model to unregister
        """
        run_out = self.connection.run(f'curl -X DELETE "http://localhost:8081/models/{model_name}/"', warn=True)
        LOGGER.info(f'curl -X DELETE "http://localhost:8081/models/{model_name}/"')
        LOGGER.info(f"stdout: {run_out.stdout}")

        time.sleep(10)
        if run_out.return_code != 0:
            LOGGER.error(f"Failed to unregister model {model_name}")


    def register_workflow(self, url):
        """
        Register an ensemble model i.e. workflow
        :param url: workflow_name of the workflow archive
        """
        run_out = self.connection.run(f'curl -X POST "http://localhost:8081/workflows?url={url}.war"')

        LOGGER.info(f'curl -X POST "http://localhost:8081/workflows?url={url}.war"')

        time.sleep(40)

    def unregister_workflow(self, workflow_name):
        """
        Deletes a workflow from the server
        :param workflow_name: name of the workflow archive
        """
        run_out = self.connection.run(f'curl -X DELETE "http://localhost:8081/workflows/{workflow_name}"')

        LOGGER.info(f'curl -X DELETE "http://localhost:8081/workflows/{workflow_name}"')


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
                activation_command = f"cd /home/ubuntu/serve/benchmark/automated/tests/resources/neuron-bert && source activate {virtual_env_name} && "
            self.connection.run(f"{activation_command}torchserve --stop", warn=True)

        time.sleep(5)
    
    def create_and_update_workflow_archive(self, workflow_name, spec_file_name, handler_file_name, batch_size, workers, batch_delay, retry_attempts, timeout_ms):
        """
        Creates the first workflow archive based on the information passed, sourced from the user-provided specfile.
        Note: the archive is created on the ec2 instance, but the specfile is modified on the current instance from which
        the benchmarks are triggered. S3 is used as a temporary storage.
        :param workflow_name: name of the workflow archive
        :param spec_file_template_path: path of the workflow specification file
        :param handler_path: path of the workflow handler file
        :param retry_attempts: retry_attempts to modify in the spec_file
        :param timeout_ms: timeout in milliseconds to be modified in the spec_file
        :param batch_delay: batch_delay in milliseconds to be modified in the spec_file
        """
        temp_uuid = uuid.uuid4()
        
        spec_file_template_path = os.path.join(self.tmp_wf_dir, spec_file_name)
        
        # Upload to s3 and fetch back to local instance: more reliable than using self.connection.get()
        self.connection.run(f"aws s3 cp {spec_file_template_path} {S3_BUCKET_BENCHMARK_ARTIFACTS}/{temp_uuid}/template.yaml")
        time.sleep(2)
        run(f"aws s3 cp {S3_BUCKET_BENCHMARK_ARTIFACTS}/{temp_uuid}/template.yaml template.yaml")
        time.sleep(2)
        
        with open("template.yaml", "r") as f:
            spec_file_contents = f.read()
            
        spec_file_contents = re.sub(r"min-workers:(.*)", f"min-workers: {workers}", spec_file_contents) 
        spec_file_contents = re.sub(r"max-workers:(.*)", f"max-workers: {workers}", spec_file_contents)
        spec_file_contents = re.sub(r"batch-size:(.*)", f"batch-size: {batch_size}", spec_file_contents)
        spec_file_contents = re.sub(r"max-batch-delay:(.*)", f"max-batch-delay: {batch_delay}", spec_file_contents)
        spec_file_contents = re.sub(r"retry-attempts:(.*)", f"retry-attempts: {retry_attempts}", spec_file_contents)
        spec_file_contents = re.sub(r"timeout-ms:(.*)", f"timeout-ms: {timeout_ms}", spec_file_contents)

        with open("template.yaml", "w") as f:
            f.write(spec_file_contents)

        run(f"aws s3 cp template.yaml {S3_BUCKET_BENCHMARK_ARTIFACTS}/{temp_uuid}/template.yaml")
        time.sleep(2)
        self.connection.run(f"aws s3 cp {S3_BUCKET_BENCHMARK_ARTIFACTS}/{temp_uuid}/template.yaml {spec_file_template_path}")

        # Clean up right away
        run(f"aws s3 rm --recursive {S3_BUCKET_BENCHMARK_ARTIFACTS}/{temp_uuid}/")
        run(f"rm template.yaml")
        
        handler_path = os.path.join(self.tmp_wf_dir, handler_file_name)

        # Note: this command places directly places the workflow archive in the wf_store
        self.connection.run(f"source activate python3 && pip install torch-workflow-archiver", shell=True)
        run_out = self.connection.run(f"/home/ubuntu/.local/bin/torch-workflow-archiver -f --workflow-name {workflow_name} --spec-file {spec_file_template_path} --handler {handler_path} --export-path /home/ubuntu/benchmark/wf_store/", shell=True)
        if run_out.return_code != 0:
            LOGGER.error(f"{run_out.stdout}")
        LOGGER.info(f"/home/ubuntu/.local/bin/torch-workflow-archiver -f --workflow-name {workflow_name} --spec-file {spec_file_template_path} --handler {handler_path} --export-path /home/ubuntu/benchmark/wf_store/")

        # Copy the *.mar files into model_store
        self.connection.run(f"cp -R {self.tmp_wf_dir}/*.mar /home/ubuntu/benchmark/model_store/", warn=True)

        LOGGER.info(f"Updated workflow archive at location: /home/ubuntu/benchmark/wf_store/{workflow_name}.war")
        
    
    def download_workflow_artifacts(self, workflow_name, model_urls, specfile_url, workflow_handler_url):
        """
        Sets up the workflow archive in the workflow store that torchserve uses. Download the artifacts into a 
        temporary folder
        :param workflow_name: name of the workflow archive
        :param model_urls: list of model urls that the workflow uses
        :param specfile_url: url of the workflow specfile
        :param workflow_handler_url: url of the handler file used by the workflow
        """
        for model_url in model_urls:
            self.connection.run(f"wget -P {self.tmp_wf_dir} {model_url}")
        
        self.connection.run(f"wget -P {self.tmp_wf_dir} {specfile_url}")
        self.connection.run(f"wget -P {self.tmp_wf_dir} {workflow_handler_url}")

        LOGGER.info(f"Downloaded workflow artifacts in the folder: {self.tmp_wf_dir}")
    

def delete_mar_file_from_model_store(model_store=None, model_mar=None):
    model_store = model_store if (model_store is not None) else f"{ROOT_DIR}/model_store/"
    if model_mar is not None:
        for f in glob.glob(os.path.join(model_store, model_mar + "*")):
            os.remove(f)
