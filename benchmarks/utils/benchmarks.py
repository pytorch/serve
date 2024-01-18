import os
import shutil
from abc import ABC, abstractmethod

import click
from utils.common import execute, is_workflow
from utils.reporting import generate_report


def create_benchmark(execution_params):
    return ABBenchmark(execution_params)


class Benchmark(ABC):
    @abstractmethod
    def warm_up():
        raise NotImplementedError

    @abstractmethod
    def run():
        raise NotImplementedError

    @abstractmethod
    def generate_report():
        raise NotImplementedError

    def prepare_environment(self):
        input = self.execution_params["input"]
        shutil.rmtree(
            os.path.join(self.execution_params["tmp_dir"], "benchmark"),
            ignore_errors=True,
        )
        shutil.rmtree(
            os.path.join(self.execution_params["report_location"], "benchmark"),
            ignore_errors=True,
        )
        os.makedirs(
            os.path.join(self.execution_params["tmp_dir"], "benchmark", "conf"),
            exist_ok=True,
        )
        os.makedirs(
            os.path.join(self.execution_params["tmp_dir"], "benchmark", "logs"),
            exist_ok=True,
        )
        os.makedirs(
            os.path.join(self.execution_params["report_location"], "benchmark"),
            exist_ok=True,
        )

        shutil.copy(
            self.execution_params["config_properties"],
            os.path.join(self.execution_params["tmp_dir"], "benchmark", "conf"),
        )
        shutil.copyfile(
            input, os.path.join(self.execution_params["tmp_dir"], "benchmark", "input")
        )


class ABBenchmark(Benchmark):
    def __init__(self, execution_params):
        self.execution_params = execution_params
        self.warm_up_lines = 0

    def warm_up(self):
        if is_workflow(self.execution_params["url"]):
            self.execution_params["inference_model_url"] = "wfpredict/benchmark"

        click.secho("\n\nExecuting warm-up ...", fg="green")

        ab_cmd = (
            f"ab -c {self.execution_params['concurrency']} -s 300 -n {self.execution_params['requests']/10} -k -p "
            f"{self.execution_params['tmp_dir']}/benchmark/input -T  {self.execution_params['content_type']} "
            f"{self.execution_params['inference_url']}/{self.execution_params['inference_model_url']} > "
            f"{self.execution_params['result_file']}"
        )
        execute(ab_cmd, wait=True)

        self.warm_up_lines = sum(1 for _ in open(self.execution_params["metric_log"]))

    def run(self):
        if is_workflow(self.execution_params["url"]):
            self.execution_params["inference_model_url"] = "wfpredict/benchmark"

        click.secho("\n\nExecuting inference performance tests ...", fg="green")
        ab_cmd = (
            f"ab -c {self.execution_params['concurrency']} -s 300 -n {self.execution_params['requests']} -k -p "
            f"{self.execution_params['tmp_dir']}/benchmark/input -T  {self.execution_params['content_type']} "
            f"{self.execution_params['inference_url']}/{self.execution_params['inference_model_url']} > "
            f"{self.execution_params['result_file']}"
        )
        execute(ab_cmd, wait=True)

    def generate_report(self):
        generate_report(self.execution_params, self.warm_up_lines)
