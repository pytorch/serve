import json
import os
import tempfile

import click
import click_config_file
from utils.benchmarks import create_benchmark
from utils.system_under_test import create_system_under_test
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
@click.option(
    "--benchmark_backend",
    "-bb",
    type=click.Choice(["ab", "locust"], case_sensitive=False),
    default="ab",
    help=f"Benchmark backend to use.",
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

    torchserve = create_system_under_test(execution_params)

    benchmark = create_benchmark(execution_params)

    benchmark.prepare_environment()

    torchserve.start()
    torchserve.check_health()
    torchserve.register_model()

    benchmark.warm_up()
    benchmark.run()

    torchserve.unregister_model()
    torchserve.stop()
    click.secho("Apache Bench Execution completed.", fg="green")

    benchmark.generate_report()
    click.secho("\nTest suite execution complete.", fg="green")


def update_exec_params(execution_params, input_param):
    execution_params.update(input_param)

    execution_params["result_file"] = os.path.join(
        execution_params["tmp_dir"], "benchmark", "result.txt"
    )
    execution_params["metric_log"] = os.path.join(
        execution_params["tmp_dir"], "benchmark", "logs", "model_metrics.log"
    )

    getAPIS(execution_params)


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


if __name__ == "__main__":
    benchmark()
