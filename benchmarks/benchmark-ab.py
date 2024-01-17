import json
import os
import shutil
import sys
import tempfile
import time
from abc import ABC, abstractmethod
from subprocess import PIPE, STDOUT, Popen
from urllib.parse import urlparse

import click
import click_config_file
import requests
from utils.reporting import generate_report
from utils.testplans import update_plan_params


def json_provider(file_path, cmd_name):
    with open(file_path) as config_data:
        return json.load(config_data)


@click.command()
@click.argument("test_plan", default="custom")
@click.option(
    "--url",
    "-u",
    default="https://torchserve.pytorch.org/mar_files/resnet-18.mar",
    help="Input model url",
)
@click.option(
    "--exec_env",
    "-e",
    type=click.Choice(["local", "docker"], case_sensitive=False),
    default="local",
    help="Execution environment",
)
@click.option(
    "--gpus",
    "-g",
    default="",
    help="Number of gpus to run docker container with.  Leave empty to run CPU based docker container",
)
@click.option(
    "--concurrency", "-c", default=10, help="Number of concurrent requests to run"
)
@click.option("--requests", "-r", default=100, help="Number of requests")
@click.option("--batch_size", "-bs", default=1, help="Batch size of model")
@click.option("--batch_delay", "-bd", default=200, help="Batch delay of model")
@click.option(
    "--input",
    "-i",
    default="../examples/image_classifier/kitten.jpg",
    type=click.Path(exists=True),
    help="The input file path for model",
)
@click.option(
    "--content_type", "-ic", default="application/jpg", help="Input file content type"
)
@click.option("--workers", "-w", default=1, help="Number model workers")
@click.option(
    "--image", "-di", default="", help="Use custom docker image for benchmark"
)
@click.option(
    "--docker_runtime", "-dr", default="", help="Specify required docker runtime"
)
@click.option(
    "--backend_profiling",
    "-bp",
    default=False,
    help="Enable backend profiling using CProfile. Default False",
)
@click.option(
    "--handler_profiling",
    "-hp",
    default=False,
    help="Enable handler profiling. Default False",
)
@click.option(
    "--generate_graphs",
    "-gg",
    default=False,
    help="Enable generation of Graph plots. Default False",
)
@click.option(
    "--config_properties",
    "-cp",
    default="config.properties",
    help="config.properties path, Default config.properties",
)
@click.option(
    "--inference_model_url",
    "-imu",
    default="predictions/benchmark",
    help="Inference function url - can be either for predictions or explanations. Default predictions/benchmark",
)
@click.option(
    "--report_location",
    "-rl",
    default=tempfile.gettempdir(),
    help=f"Target location of benchmark report. Default {tempfile.gettempdir()}",
)
@click.option(
    "--tmp_dir",
    "-td",
    default=tempfile.gettempdir(),
    help=f"Location for temporal files. Default {tempfile.gettempdir()}",
)
@click_config_file.configuration_option(
    provider=json_provider, implicit=False, help="Read configuration from a JSON file"
)
def benchmark(test_plan, **input_params):
    execution_params = input_params.copy()

    # set ab params
    update_plan_params[test_plan](execution_params)
    update_exec_params(execution_params, input_params)

    click.secho("Starting AB benchmark suite...", fg="green")
    click.secho("\n\nConfigured execution parameters are:", fg="green")
    click.secho(f"{execution_params}", fg="blue")

    prepare_common_dependency(execution_params)

    torchserve = create_system_under_test(execution_params)

    torchserve.start()

    torchserve.check_health()

    torchserve.register_model()

    warm_up_lines = warm_up(execution_params)
    run_benchmark(execution_params)

    torchserve.unregister_model()

    torchserve.stop()
    click.secho("Apache Bench Execution completed.", fg="green")

    generate_report(execution_params, warm_up_lines=warm_up_lines)


def warm_up(execution_params):
    if is_workflow(execution_params["url"]):
        execution_params["inference_model_url"] = "wfpredict/benchmark"

    click.secho("\n\nExecuting warm-up ...", fg="green")

    ab_cmd = (
        f"ab -c {execution_params['concurrency']} -s 300 -n {execution_params['requests']/10} -k -p "
        f"{execution_params['tmp_dir']}/benchmark/input -T  {execution_params['content_type']} "
        f"{execution_params['inference_url']}/{execution_params['inference_model_url']} > "
        f"{execution_params['result_file']}"
    )
    execute(ab_cmd, wait=True)

    warm_up_lines = sum(1 for _ in open(execution_params["metric_log"]))

    return warm_up_lines


def run_benchmark(execution_params):
    if is_workflow(execution_params["url"]):
        execution_params["inference_model_url"] = "wfpredict/benchmark"

    click.secho("\n\nExecuting inference performance tests ...", fg="green")
    ab_cmd = (
        f"ab -c {execution_params['concurrency']} -s 300 -n {execution_params['requests']} -k -p "
        f"{execution_params['tmp_dir']}/benchmark/input -T  {execution_params['content_type']} "
        f"{execution_params['inference_url']}/{execution_params['inference_model_url']} > "
        f"{execution_params['result_file']}"
    )
    execute(ab_cmd, wait=True)


def execute(command, wait=False, stdout=None, stderr=None, shell=True):
    print(command)
    cmd = Popen(
        command,
        shell=shell,
        close_fds=True,
        stdout=stdout,
        stderr=stderr,
        universal_newlines=True,
    )
    if wait:
        cmd.wait()
    return cmd


def execute_return_stdout(cmd):
    proc = execute(cmd, stdout=PIPE)
    return proc.communicate()[0].strip()


def prepare_common_dependency(execution_params):
    input = execution_params["input"]
    shutil.rmtree(
        os.path.join(execution_params["tmp_dir"], "benchmark"), ignore_errors=True
    )
    shutil.rmtree(
        os.path.join(execution_params["report_location"], "benchmark"),
        ignore_errors=True,
    )
    os.makedirs(
        os.path.join(execution_params["tmp_dir"], "benchmark", "conf"), exist_ok=True
    )
    os.makedirs(
        os.path.join(execution_params["tmp_dir"], "benchmark", "logs"), exist_ok=True
    )
    os.makedirs(
        os.path.join(execution_params["report_location"], "benchmark"), exist_ok=True
    )

    shutil.copy(
        execution_params["config_properties"],
        os.path.join(execution_params["tmp_dir"], "benchmark", "conf"),
    )
    shutil.copyfile(
        input, os.path.join(execution_params["tmp_dir"], "benchmark", "input")
    )


def getAPIS(execution_params):
    MANAGEMENT_API = "http://127.0.0.1:8081"
    INFERENCE_API = "http://127.0.0.1:8080"

    with open(execution_params["config_properties"], "r") as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.strip()
        if "management_address" in line:
            MANAGEMENT_API = line.split("=")[1]
        if "inference_address" in line:
            INFERENCE_API = line.split("=")[1]

    execution_params["inference_url"] = INFERENCE_API
    execution_params["management_url"] = MANAGEMENT_API
    execution_params["config_properties_name"] = (
        execution_params["config_properties"].strip().split("/")[-1]
    )


def update_exec_params(execution_params, input_param):
    execution_params.update(input_param)

    execution_params["result_file"] = os.path.join(
        execution_params["tmp_dir"], "benchmark", "result.txt"
    )
    execution_params["metric_log"] = os.path.join(
        execution_params["tmp_dir"], "benchmark", "logs", "model_metrics.log"
    )

    getAPIS(execution_params)


def create_system_under_test(execution_params):
    # Setup execution env
    if execution_params["exec_env"] == "local":
        click.secho("\n\nPreparing local execution...", fg="green")
        return LocalTorchServeUnderTest(execution_params)
    else:
        click.secho("\n\nPreparing docker execution...", fg="green")
        return DockerTorchServeUnderTest(execution_params)


class SystemUnderTest(ABC):
    def __init__(self):
        pass

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
        check_torchserve_health(self.execution_params)


class LocalTorchServeUnderTest(TorchServeUnderTest):
    def start(self):
        click.secho("*Terminating any existing Torchserve instance ...", fg="green")
        execute("torchserve --stop", wait=True)
        click.secho("*Setting up model store...", fg="green")
        self._prepare_local_dependency()
        click.secho("*Starting local Torchserve instance...", fg="green")

        ts_cmd = (
            f"torchserve --start --model-store {self.execution_params['tmp_dir']}/model_store "
            f"--workflow-store {self.execution_params['tmp_dir']}/wf_store "
            f"--ts-config {self.execution_params['tmp_dir']}/benchmark/conf/{self.execution_params['config_properties_name']} "
        )

        tee_cmd = (
            f"tee {self.execution_params['tmp_dir']}/benchmark/logs/model_metrics.log"
        )
        click.secho(f"Running: {ts_cmd} | {tee_cmd}")
        from shlex import split

        ts_p = Popen(split(ts_cmd), stdin=PIPE, stdout=PIPE, stderr=STDOUT)
        tee_p = Popen(split(tee_cmd), stdin=ts_p.stdout, stdout=PIPE, stderr=STDOUT)

        for line in tee_p.stdout:
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
            f'"torchserve --start --model-store /home/model-server/model-store '
            f"\--workflow-store /home/model-server/wf-store "
            f"--ts-config /tmp/benchmark/conf/{self.execution_params['config_properties_name']} > "
            f'/tmp/benchmark/logs/model_metrics.log"'
        )
        execute(docker_run_cmd, wait=True)
        time.sleep(5)

    def stop(self):
        click.secho("*Removing benchmark container 'ts'...", fg="green")
        execute("docker rm -f ts", wait=True)


def check_torchserve_health(execution_params):
    attempts = 3
    retry = 0
    click.secho("*Testing system health...", fg="green")
    while retry < attempts:
        try:
            resp = requests.get(execution_params["inference_url"] + "/ping")
            if resp.status_code == 200:
                click.secho(resp.text)
                return True
        except Exception as e:
            retry += 1
            time.sleep(3)
    failure_exit(
        "Could not connect to Torchserve instance at "
        + execution_params["inference_url"]
    )


def failure_exit(msg):
    click.secho(f"{msg}", fg="red")
    click.secho("Test suite terminated due to above failure", fg="red")
    sys.exit()


def is_workflow(model_url):
    return model_url.endswith(".war")


if __name__ == "__main__":
    benchmark()
