import argparse
import copy
import os

import yaml
from auto_benchmark import load_benchmark_config
from utils.gen_model_config_json import MODEL_CONFIG_KEY
from utils.report import Report

BENCHMARK_REPORT_CSV = "ab_report.csv"
CWD = os.getcwd()
VALIDATION_KEYS = [
    "throughput",
    "total_latency_p50",
    "model_latency_p50",
    "total_latency_p90",
    "model_latency_p90",
    "total_latency_p99",
    "model_latency_p99",
]


def conv_model_yaml_dict(model_file):

    with open(model_file, "r") as f:
        yaml_dict = yaml.safe_load(f)

        for model, config in yaml_dict.items():
            benchmark_configs = []
            for mode, mode_config in config.items():
                model_name = mode + "_" + model
                benchmark_config = {}
                batch_size_list = None
                processors = None
                workers_list = None
                benchmark_config["model"] = model
                benchmark_config["mode"] = mode
                for key, value in mode_config.items():
                    if key == "batch_size":
                        batch_size_list = value
                    elif key == "processors":
                        processors = value
                    elif key == "workers":
                        workers_list = value
                    elif key in MODEL_CONFIG_KEY:
                        benchmark_config[key] = value

                batch_worker_list = []
                for batch_size in batch_size_list:
                    for workers in workers_list:
                        batch_worker_list.append(
                            {"batch_size": batch_size, "workers": workers}
                        )

                for batch_worker in batch_worker_list:
                    benchmark_config["batch_size"] = batch_worker["batch_size"]
                    benchmark_config["workers"] = batch_worker["workers"]
                    benchmark_configs.append(copy.deepcopy(benchmark_config))

    return benchmark_configs


def check_if_within_range(value1, value2, threshold):
    return abs((value1 - value2) / float(value1)) <= threshold


def validate_reports(args):
    input_dir = args.input
    if not os.path.isdir(input_dir):
        print("No report generated")
        return -1

    # Read baseline reports
    bm_config = load_benchmark_config(args.input_cfg, True, True)
    baseline_reports = {}
    for model in bm_config["models"]:
        model_file = CWD + "/benchmarks/models_config/{}".format(model)
        benchmark_configs = conv_model_yaml_dict(model_file)

        for config in benchmark_configs:
            yaml_file = CWD + "/benchmarks/models_baseline/{}".format(model)
            report = Report()
            report.read_yaml(yaml_file, config)
            key = (
                config["mode"]
                + "_"
                + config["model"]
                + "_w"
                + str(config["workers"])
                + "_b"
                + str(config["batch_size"])
            )
            baseline_reports[key] = report
    print("BAseline !!!!! ", baseline_reports)
    # Read generated reports
    generated_reports = {}
    for subdir in sorted(os.listdir(input_dir)):
        if os.path.isdir(os.path.join(input_dir, subdir)):
            csv_file = os.path.join(input_dir, subdir, BENCHMARK_REPORT_CSV)
            report = Report()
            report.read_csv(csv_file)
            generated_reports[subdir] = report
    print("Generated ##### ", generated_reports)

    for model, report in generated_reports.items():
        mode = report.mode
        for key in VALIDATION_KEYS:
            if not check_if_within_range(
                report.properties[key],
                baseline_reports[model].properties[mode][key],
                baseline_reports[model].properties[mode]["deviation"],
            ):
                print(
                    f"Error while validating {key} for model: {model}, "
                    f"Expected value: {baseline_reports[model].properties[mode][key]},"
                    f"Observed value: {report.properties[key]}"
                )
                return -1
        print(f"Model {model} successfully validated")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        action="store",
        help="the dir of a list of model benchmark result subdir ",
    )

    parser.add_argument(
        "--input_cfg",
        action="store",
        help="benchmark config yaml file path",
    )

    parser.add_argument(
        "--output",
        action="store",
        help="the file path of final report ",
    )

    arguments = parser.parse_args()
    validate_reports(arguments)


if __name__ == "__main__":
    main()
