import csv
import os
import re

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def generate_report(execution_params, warm_up_lines):
    click.secho("\n\nGenerating Reports...", fg="green")
    extract_metrics(execution_params, warm_up_lines=warm_up_lines)
    generate_csv_output(execution_params)
    if execution_params["generate_graphs"]:
        generate_latency_graph(execution_params)
        generate_profile_graph(execution_params)


metrics = {
    "predict.txt": "PredictionTime",
    "handler_time.txt": "HandlerTime",
    "waiting_time.txt": "QueueTime",
    "worker_thread.txt": "WorkerThreadTime",
    "cpu_percentage.txt": "CPUUtilization",
    "memory_percentage.txt": "MemoryUtilization",
    "gpu_percentage.txt": "GPUUtilization",
    "gpu_memory_percentage.txt": "GPUMemoryUtilization",
    "gpu_memory_used.txt": "GPUMemoryUsed",
}


def update_metrics(execution_params):
    if execution_params["handler_profiling"]:
        opt_metrics = {
            "handler_preprocess.txt": "ts_handler_preprocess",
            "handler_inference.txt": "ts_handler_inference",
            "handler_postprocess.txt": "ts_handler_postprocess",
        }
        metrics.update(opt_metrics)
    return metrics


def extract_metrics(execution_params, warm_up_lines):
    with open(execution_params["metric_log"]) as f:
        lines = f.readlines()

    click.secho(f"Dropping {warm_up_lines} warmup lines from log", fg="green")
    lines = lines[warm_up_lines:]

    metrics = update_metrics(execution_params)

    for k, v in metrics.items():
        all_lines = []
        pattern = re.compile(v)
        for line in lines:
            if pattern.search(line):
                all_lines.append(line.split("|")[0].split(":")[3].strip())

        out_fname = os.path.join(*(execution_params["tmp_dir"], "benchmark", k))
        click.secho(f"\nWriting extracted {v} metrics to {out_fname} ", fg="green")
        with open(out_fname, "w") as outf:
            all_lines = map(lambda x: x + "\n", all_lines)
            outf.writelines(all_lines)


def generate_csv_output(execution_params):
    click.secho("*Generating CSV output...", fg="green")
    batched_requests = execution_params["requests"] / execution_params["batch_size"]
    line50 = int(batched_requests / 2)
    line90 = int(batched_requests * 9 / 10)
    line99 = int(batched_requests * 99 / 100)

    click.secho(f"Saving benchmark results to {execution_params['report_location']}")

    artifacts = {}
    with open(execution_params["result_file"]) as f:
        data = f.readlines()

    artifacts["Benchmark"] = "AB"
    artifacts["Batch size"] = execution_params["batch_size"]
    artifacts["Batch delay"] = execution_params["batch_delay"]
    artifacts["Workers"] = execution_params["workers"]
    artifacts["Model"] = "[.mar]({})".format(execution_params["url"])
    artifacts["Concurrency"] = execution_params["concurrency"]
    artifacts["Input"] = "[input]({})".format(execution_params["input"])
    artifacts["Requests"] = execution_params["requests"]
    artifacts["TS failed requests"] = extract_entity(data, "Failed requests:", -1)
    artifacts["TS throughput"] = extract_entity(data, "Requests per second:", -3)
    artifacts["TS latency P50"] = extract_entity(data, "50%", -1)
    artifacts["TS latency P90"] = extract_entity(data, "90%", -1)
    artifacts["TS latency P99"] = extract_entity(data, "99%", -1)
    artifacts["TS latency mean"] = extract_entity(data, "Time per request:.*mean\)", -3)
    if isinstance(artifacts["TS failed requests"], type(None)):
        artifacts["TS error rate"] = 0.0
    else:
        artifacts["TS error rate"] = (
            int(artifacts["TS failed requests"]) / execution_params["requests"] * 100
        )

    with open(
        os.path.join(execution_params["tmp_dir"], "benchmark", "predict.txt")
    ) as f:
        lines = f.readlines()
        lines.sort(key=float)
        artifacts["Model_p50"] = lines[line50].strip()
        artifacts["Model_p90"] = lines[line90].strip()
        artifacts["Model_p99"] = lines[line99].strip()

    with open(
        os.path.join(execution_params["tmp_dir"], "benchmark", "waiting_time.txt")
    ) as f:
        lines = f.readlines()
        lines.sort(key=float)
        num_requests = len(lines)
        line50 = int(num_requests / 2)
        line90 = int(num_requests * 9 / 10)
        line99 = int(num_requests * 99 / 100)
        artifacts["Queue time p50"] = lines[line50].strip()
        artifacts["Queue time p90"] = lines[line90].strip()
        artifacts["Queue time p99"] = lines[line99].strip()

    for m in metrics:
        df = pd.read_csv(
            os.path.join(*(execution_params["tmp_dir"], "benchmark", m)),
            header=None,
            names=["data"],
        )
        if df.empty:
            artifacts[m.split(".txt")[0] + "_mean"] = 0.0
        else:
            artifacts[m.split(".txt")[0] + "_mean"] = df["data"].values.mean().round(2)

    with open(
        os.path.join(execution_params["report_location"], "benchmark", "ab_report.csv"),
        "w",
    ) as csv_file:
        csvwriter = csv.writer(csv_file)
        csvwriter.writerow(artifacts.keys())
        csvwriter.writerow(artifacts.values())

    return artifacts


def extract_entity(data, pattern, index, delim=" "):
    pattern = re.compile(pattern)
    for line in data:
        if pattern.search(line):
            return line.split(delim)[index].strip()
    return None


def generate_latency_graph(execution_params):
    click.secho("*Preparing graphs...", fg="green")
    df = pd.read_csv(
        os.path.join(execution_params["tmp_dir"], "benchmark", "predict.txt"),
        header=None,
        names=["latency"],
    )
    iteration = df.index
    latency = df.latency
    a4_dims = (11.7, 8.27)
    plt.figure(figsize=(a4_dims))
    plt.xlabel("Requests")
    plt.ylabel("Prediction time")
    plt.title("Prediction latency")
    plt.bar(iteration, latency)
    plt.savefig(f"{execution_params['report_location']}/benchmark/predict_latency.png")


def generate_profile_graph(execution_params):
    click.secho("*Preparing Profile graphs...", fg="green")

    plot_data = {}
    for m in metrics:
        file_path = f"{execution_params['tmp_dir']}/benchmark/{m}"
        if is_file_empty(file_path):
            continue
        df = pd.read_csv(file_path, header=None)
        m = m.split(".txt")[0]
        plot_data[f"{m}_index"] = df.index
        plot_data[f"{m}_values"] = df.values

    if execution_params["requests"] > 100:
        sampling = int(execution_params["requests"] / 100)
    else:
        sampling = 1
    print(f"Working with sampling rate of {sampling}")

    a4_dims = (11.7, 8.27)
    grid = plt.GridSpec(3, 2, wspace=0.2, hspace=0.2)
    plt.figure(figsize=a4_dims)
    fig1 = plt.subplot(grid[0, 0])
    fig2 = plt.subplot(grid[0, 1])
    fig3 = plt.subplot(grid[1, 0])
    fig4 = plt.subplot(grid[1, 1])
    fig5 = plt.subplot(grid[2, 0:])

    def plot_line(fig, data, color="blue", title=None):
        fig.set_title(title)
        fig.set_ylabel("Time (ms)")
        fig.set_xlabel("Percentage of queries")
        fig.grid()
        plot_points = np.arange(0, 100, 100 / len(data))
        x = plot_points[: len(data) : sampling]
        y = data[::sampling]
        fig.plot(x, y, f"tab:{color}")

    # Queue Time
    plot_line(
        fig1, data=plot_data["waiting_time_values"], color="pink", title="Queue Time"
    )

    # handler Predict Time
    plot_line(
        fig2,
        data=plot_data["handler_time_values"],
        color="orange",
        title="Handler Time(pre & post processing + inference time)",
    )

    # Worker time
    plot_line(
        fig3,
        data=plot_data["worker_thread_values"],
        color="green",
        title="Worker Thread Time",
    )

    # Predict Time
    plot_line(
        fig4,
        data=plot_data["predict_values"],
        color="red",
        title="Prediction time(handler time+python worker overhead)",
    )

    # Plot in one graph
    plot_line(fig5, data=plot_data["waiting_time_values"], color="pink")
    plot_line(fig5, data=plot_data["handler_time_values"], color="orange")
    plot_line(fig5, data=plot_data["predict_values"], color="red")
    plot_line(
        fig5,
        data=plot_data["worker_thread_values"],
        color="green",
        title="Combined Graph",
    )
    fig5.grid()
    plt.savefig(
        f"{execution_params['report_location']}/benchmark/api-profile1.png",
        bbox_inches="tight",
    )

    def is_file_empty(file_path):
        """Check if file is empty by confirming if its size is 0 bytes"""
        # Check if file exist and it is empty
        return os.path.exists(file_path) and os.stat(file_path).st_size == 0
