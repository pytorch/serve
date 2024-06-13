import os
import shutil
import sys
import time
from abc import ABC, abstractmethod
from pathlib import Path
from urllib.parse import urlparse

import click
import requests
from utils.common import execute, is_workflow


def create_system_under_test(execution_params):
    # Setup execution env
    if execution_params["exec_env"] == "local":
        click.secho("\n\nPreparing local execution...", fg="green")
        return LocalTorchServeUnderTest(execution_params)
    else:
        click.secho("\n\nPreparing docker execution...", fg="green")
        return DockerTorchServeUnderTest(execution_params)


class SystemUnderTest(ABC):
    @abstractmethod
    def start(self):
        raise NotImplementedError

    @abstractmethod
    def register_model(self):
        raise NotImplementedError

    @abstractmethod
    def unregister_model(self):
        raise NotImplementedError

    @abstractmethod
    def check_health(self):
        raise NotImplementedError

    @abstractmethod
    def stop(self):
        raise NotImplementedError


class TorchServeUnderTest(SystemUnderTest):
    def __init__(self, execution_params):
        self.execution_params = execution_params

    def register_model(self):
        click.secho("*Registering model...", fg="green")
        if is_workflow(self.execution_params["url"]):
            url = self.execution_params["management_url"] + "/workflows"
            data = {
                "workflow_name": "benchmark",
                "url": self.execution_params["url"],
                "batch_delay": self.execution_params["batch_delay"],
                "batch_size": self.execution_params["batch_size"],
                "initial_workers": self.execution_params["workers"],
                "synchronous": "true",
            }
        else:
            url = self.execution_params["management_url"] + "/models"
            data = {
                "model_name": "benchmark",
                "url": self.execution_params["url"],
                "batch_delay": self.execution_params["batch_delay"],
                "batch_size": self.execution_params["batch_size"],
                "initial_workers": self.execution_params["workers"],
                "synchronous": "true",
            }
        resp = requests.post(url, params=data)
        if not resp.status_code == 200:
            failure_exit(f"Failed to register model.\n{resp.text}")
        click.secho(resp.text)

    def unregister_model(self):
        click.secho("*Unregistering model ...", fg="green")
        if is_workflow(self.execution_params["url"]):
            resp = requests.delete(
                self.execution_params["management_url"] + "/workflows/benchmark"
            )
        else:
            resp = requests.delete(
                self.execution_params["management_url"] + "/models/benchmark"
            )
        if not resp.status_code == 200:
            failure_exit(f"Failed to unregister model. \n {resp.text}")
        click.secho(resp.text)

    def check_health(self):
        attempts = 3
        retry = 0
        click.secho("*Testing system health...", fg="green")
        while retry < attempts:
            try:
                resp = requests.get(self.execution_params["inference_url"] + "/ping")
                if resp.status_code == 200:
                    click.secho(resp.text)
                    return True
            except Exception as e:
                retry += 1
                time.sleep(3)
        failure_exit(
            "Could not connect to Torchserve instance at "
            + self.execution_params["inference_url"]
        )


class LocalTorchServeUnderTest(TorchServeUnderTest):
    def start(self):
        click.secho("*Terminating any existing Torchserve instance ...", fg="green")
        execute("torchserve --stop", wait=True)
        click.secho("*Setting up model store...", fg="green")
        self._prepare_local_dependency()
        click.secho("*Starting local Torchserve instance...", fg="green")

        ts_cmd = (
            f"torchserve --start --model-store {self.execution_params['tmp_dir']}/model_store --model-api-enabled --disable-token "
            f"--workflow-store {self.execution_params['tmp_dir']}/wf_store "
            f"--ts-config {self.execution_params['tmp_dir']}/benchmark/conf/{self.execution_params['config_properties_name']} "
            f" > {self.execution_params['tmp_dir']}/benchmark/logs/model_metrics.log"
        )

        click.secho(f"Running: {ts_cmd}")
        execute(ts_cmd)
        n = 0
        while (
            not Path(
                f"{self.execution_params['tmp_dir']}/benchmark/logs/model_metrics.log"
            ).exists()
            and n < 30
        ):
            time.sleep(0.1)
            n += 1

        with open(
            f"{self.execution_params['tmp_dir']}/benchmark/logs/model_metrics.log", "r"
        ) as f:
            for line in f.readlines():
                if "Model server started" in str(line).strip():
                    break

    def stop(self):
        click.secho("*Terminating Torchserve instance...", fg="green")
        execute("torchserve --stop", wait=True)

    def _prepare_local_dependency(self):
        shutil.rmtree(
            os.path.join(self.execution_params["tmp_dir"], "model_store/"),
            ignore_errors=True,
        )
        os.makedirs(
            os.path.join(self.execution_params["tmp_dir"], "model_store/"),
            exist_ok=True,
        )
        shutil.rmtree(
            os.path.join(self.execution_params["tmp_dir"], "wf_store/"),
            ignore_errors=True,
        )
        os.makedirs(
            os.path.join(self.execution_params["tmp_dir"], "wf_store/"), exist_ok=True
        )


class DockerTorchServeUnderTest(TorchServeUnderTest):
    def start(self):
        enable_gpu = ""
        if self.execution_params["image"]:
            docker_image = self.execution_params["image"]
            if self.execution_params["gpus"]:
                enable_gpu = f"--gpus {self.execution_params['gpus']}"
        else:
            if self.execution_params["gpus"]:
                docker_image = "pytorch/torchserve:latest-gpu"
                enable_gpu = f"--gpus {self.execution_params['gpus']}"
            else:
                docker_image = "pytorch/torchserve:latest"
            execute(f"docker pull {docker_image}", wait=True)

        backend_profiling = ""
        if self.execution_params["backend_profiling"]:
            backend_profiling = "-e TS_BENCHMARK=True"

        # delete existing ts container instance
        click.secho("*Removing existing ts container instance...", fg="green")
        execute("docker rm -f ts", wait=True)

        click.secho(
            f"*Starting docker container of image {docker_image} ...", fg="green"
        )
        inference_port = urlparse(self.execution_params["inference_url"]).port
        management_port = urlparse(self.execution_params["management_url"]).port
        docker_run_cmd = (
            f"docker run {self.execution_params['docker_runtime']} {backend_profiling} --name ts --user root -p "
            f"127.0.0.1:{inference_port}:{inference_port} -p 127.0.0.1:{management_port}:{management_port} "
            f"-v {self.execution_params['tmp_dir']}:/tmp {enable_gpu} -itd {docker_image} "
            f'"torchserve --start --model-store /home/model-server/model-store --model-api-enabled --disable-token '
            f"\--workflow-store /home/model-server/wf-store "
            f"--ts-config /tmp/benchmark/conf/{self.execution_params['config_properties_name']} > "
            f'/tmp/benchmark/logs/model_metrics.log"'
        )
        execute(docker_run_cmd, wait=True)
        time.sleep(5)

    def stop(self):
        click.secho("*Removing benchmark container 'ts'...", fg="green")
        execute("docker rm -f ts", wait=True)


def failure_exit(msg):
    click.secho(f"{msg}", fg="red")
    click.secho("Test suite terminated due to above failure", fg="red")
    sys.exit()
