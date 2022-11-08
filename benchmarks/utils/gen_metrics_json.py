import argparse
import csv
import json
import re

# Ref valid unit:
# https://docs.aws.amazon.com/AmazonCloudWatch/latest/APIReference/API_MetricDatum.html
UNIT_MAP = {
    "Count": "Count",
    "Milliseconds": "Milliseconds",
    "ms": "Milliseconds",
    "Megabytes": "Megabytes",
    "MB": "Megabytes",
    "Gigabytes": "Gigabytes",
    "GB": "Gigabytes",
    "Bytes": "Bytes",
    "B": "Bytes",
    "Percent": "Percent",
    "s": "Seconds",
}

METRICS_NAME_SET = {
    "GPUUtilization",
    "GPUMemoryUtilization",
    "GPUMemoryUsed",
    "CPUUtilization",
    "DiskAvailable",
    "DiskUsed",
    "DiskUtilization",
    "MemoryUsed",
    "MemoryUtilization",
    "Requests2XX",
    "Requests4XX",
    "Requests5XX",
    "PredictionTime",
    "HandlerTime",
}

STATS_METRICS_CONFIG = {
    "TS throughput": {"name": "throughput", "unit": "Count/Second"},
    "TS latency P50": {"name": "total_latency_P50", "unit": "Milliseconds"},
    "TS latency P90": {"name": "total_latency_P90", "unit": "Milliseconds"},
    "TS latency P99": {"name": "total_latency_P99", "unit": "Milliseconds"},
    "Model_p50": {"name": "model_latency_P50", "unit": "Milliseconds"},
    "Model_p90": {"name": "model_latency_P90", "unit": "Milliseconds"},
    "Model_p99": {"name": "model_latency_P99", "unit": "Milliseconds"},
    "memory_percentage_mean": {"name": "memory_percentage_mean", "unit": "Percent"},
    "cpu_percentage_mean": {"name": "cpu_percentage_mean", "unit": "Percent"},
    "gpu_percentage_mean": {"name": "gpu_percentage_mean", "unit": "Percent"},
    "gpu_percentage_mean": {"name": "gpu_percentage_mean", "unit": "Percent"},
    "gpu_memory_used_mean": {"name": "gpu_memory_used_mean", "unit": "Megabytes"},
}


# Ref metrics json format
# https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/CloudWatch-metric-streams-formats-json.html
def gen_metrics_from_csv(csv_dict, stats_file_path):
    if csv_dict is None:
        return

    metrics_dict_list = []
    for k, v in csv_dict.items():
        if k in STATS_METRICS_CONFIG and isinstance(v, (int, float)):
            metric_config = STATS_METRICS_CONFIG[k]
            metrics_dict_list.append(
                {
                    "MetricName": "{}_{}".format(
                        csv_dict["Model"], metric_config["name"]
                    ),
                    "Dimensions": [
                        {"Name": "batch_size", "Value": csv_dict["Batch size"]}
                    ],
                    "Unit": metric_config["unit"],
                    "Value": float(v),
                }
            )

    with open(stats_file_path, "w") as stats_file:
        json.dump(metrics_dict_list, stats_file, indent=4)


def extract_metrics_from_csv(csv_file_path):
    with open(csv_file_path, "r") as csvfile:
        csv_reader = csv.DictReader(csvfile, delimiter=",")
        for row in csv_reader:
            model = row["Model"]
            index = model.rfind("/") + 1
            row["Model"] = model[index:-4]
            return row

    return None


def gen_metrics_from_log(csv_dict, metrics_log_file_path, raw_metrics_file_path):
    if metrics_log_file_path is None or raw_metrics_file_path is None:
        return

    with open(metrics_log_file_path, "r") as logfile:
        lines = logfile.readlines()

    metrics_dict_list = []
    pattern = re.compile(" TS_METRICS | MODEL_METRICS ")
    for line in lines:
        if pattern.search(line):
            segments = line.split("|")
            name, unit, value = parse_segments_0(segments[0])
            if name is None:
                continue
            dimensions = parse_segments_1(csv_dict, segments[1])
            timestamp = parse_segments_2(segments[2])
            metrics_dict_list.append(
                {
                    "MetricName": "{}_{}".format(csv_dict["Model"], name),
                    "Dimensions": dimensions,
                    "Unit": unit,
                    "Value": float(value),
                    "Timestamp": timestamp,
                }
            )

    with open(raw_metrics_file_path, "w") as raw_file:
        json.dump(metrics_dict_list, raw_file, indent=4)


def parse_segments_0(segment):
    index = segment.rfind(" ") + 1
    data = segment[index:].split(":")
    value = data[1]
    name_unit = data[0].split(".")
    if name_unit[0] in METRICS_NAME_SET:
        name = name_unit[0]
        unit = UNIT_MAP[name_unit[1]]
    else:
        name = None
        unit = None

    return name, unit, value


def parse_segments_1(csv_dict, segment):
    data = segment[1:].split(",")
    dimensions = [{"Name": "batch_size", "Value": csv_dict["Batch size"]}]
    for d in data:
        dimension = d.split(":")
        dimensions.append({"Name": dimension[0], "Value": dimension[1]})

    return dimensions


def parse_segments_2(segment):
    index = segment.rfind(",") + 1
    data = segment[index:].split(":")
    return int(data[1])


def gen_metric(csv_file, stats_metrics_file):
    csv_dict = extract_metrics_from_csv(csv_file)
    gen_metrics_from_csv(csv_dict, stats_metrics_file)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csv",
        action="store",
        help="ab report csv file path",
    )

    parser.add_argument(
        "--log",
        action="store",
        help="model_metrics.log file path",
    )

    parser.add_argument(
        "--stats",
        action="store",
        help="output stats metrics json file path",
    )

    parser.add_argument(
        "--raw",
        action="store",
        help="output raw metrics json file path",
    )

    arguments = parser.parse_args()
    csv_dict = extract_metrics_from_csv(arguments.csv)
    gen_metrics_from_csv(csv_dict, arguments.stats)
    gen_metrics_from_log(csv_dict, arguments.log, arguments.raw)


if __name__ == "__main__":
    main()
