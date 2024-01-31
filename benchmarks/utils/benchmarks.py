import os
import shutil
from abc import ABC, abstractmethod
from pathlib import Path

import click
from utils.common import execute, is_workflow
from utils.reporting import (
    extract_ab_tool_benchmark_artifacts,
    extract_locust_tool_benchmark_artifacts,
    extract_metrics,
    generate_csv_output,
    generate_latency_graph,
    generate_profile_graph,
)


def create_benchmark(execution_params):
    if execution_params["benchmark_backend"] == "ab":
        return ABBenchmark(execution_params)
    else:
        return LocustBenchmark(execution_params)


class Benchmark(ABC):
    def __init__(self, execution_params):
        self.execution_params = execution_params
        self.warm_up_lines = 0
        if is_workflow(self.execution_params["url"]):
            self.execution_params["inference_model_url"] = "wfpredict/benchmark"

    @abstractmethod
    def warm_up():
        raise NotImplementedError

    @abstractmethod
    def run():
        raise NotImplementedError

    def generate_report(self):
        click.secho("\n\nGenerating Reports...", fg="green")
        metrics = extract_metrics(
            self.execution_params, warm_up_lines=self.warm_up_lines
        )

        artifacts = {}
        benchmark_artifacts = self._extract_benchmark_artifacts()
        artifacts.update(benchmark_artifacts)

        generate_csv_output(self.execution_params, metrics, benchmark_artifacts)
        if self.execution_params["generate_graphs"]:
            generate_latency_graph(self.execution_params)
            generate_profile_graph(self.execution_params, metrics)

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

    @abstractmethod
    def _extract_benchmark_artifacts(self):
        raise NotImplementedError


class ABBenchmark(Benchmark):
    def warm_up(self):
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
        click.secho("\n\nExecuting inference performance tests ...", fg="green")
        ab_cmd = (
            f"ab -c {self.execution_params['concurrency']} -s 300 -n {self.execution_params['requests']} -k -p "
            f"{self.execution_params['tmp_dir']}/benchmark/input -T  {self.execution_params['content_type']} "
            f"{self.execution_params['inference_url']}/{self.execution_params['inference_model_url']} > "
            f"{self.execution_params['result_file']}"
        )
        execute(ab_cmd, wait=True)

    def _extract_benchmark_artifacts(self):
        return extract_ab_tool_benchmark_artifacts(self.execution_params)


class LocustBenchmark(Benchmark):
    def __init__(self, execution_params):
        self.locust_benchmark_file = Path(__file__).parent / "locust_benchmark.py"
        super().__init__(execution_params)

    def warm_up(self):
        locust_cmd = (
            f"locust  -H {self.execution_params['inference_url']} --locustfile {self.locust_benchmark_file} "
            f"--headless --reset-stats -u {self.execution_params['concurrency']} -r {self.execution_params['concurrency']} -i {self.execution_params['requests']//10} "
            f"--input {self.execution_params['tmp_dir']}/benchmark/input --content-type  {self.execution_params['content_type']} "
            f"--model-url {self.execution_params['inference_model_url']} "
        )
        click.secho("\n\nExecuting warm-up ...", fg="green")

        execute(locust_cmd, wait=True)

        self.warm_up_lines = sum(1 for _ in open(self.execution_params["metric_log"]))

    def run(self):
        locust_cmd = (
            f"locust  -H {self.execution_params['inference_url']} --locustfile {self.locust_benchmark_file} "
            f"--headless --reset-stats -u {self.execution_params['concurrency']} -r {self.execution_params['concurrency']} -i {self.execution_params['requests']} "
            f"--input {self.execution_params['tmp_dir']}/benchmark/input --content-type  {self.execution_params['content_type']} "
            f"--model-url {self.execution_params['inference_model_url']} "
            f"--json > {self.execution_params['result_file']}"
        )
        click.secho("\n\nExecuting inference performance tests ...", fg="green")

        execute(locust_cmd, wait=True)

    def _extract_benchmark_artifacts(self):
        return extract_locust_tool_benchmark_artifacts(self.execution_params)
