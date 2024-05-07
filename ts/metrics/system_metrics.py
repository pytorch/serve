"""
Module to collect system metrics for front-end
"""
import logging
import types
from builtins import str
import time

import psutil
import subprocess
from ts.metrics.dimension import Dimension
from ts.metrics.metric import Metric

system_metrics = []
dimension = [Dimension("Level", "Host")]


def cpu_utilization():
    data = psutil.cpu_percent()
    system_metrics.append(Metric("CPUUtilization", data, "percent", dimension))


def memory_used():
    data = psutil.virtual_memory().used / (1024 * 1024)  # in MB
    system_metrics.append(Metric("MemoryUsed", data, "MB", dimension))


def memory_available():
    data = psutil.virtual_memory().available / (1024 * 1024)  # in MB
    system_metrics.append(Metric("MemoryAvailable", data, "MB", dimension))


def memory_utilization():
    data = psutil.virtual_memory().percent
    system_metrics.append(Metric("MemoryUtilization", data, "percent", dimension))


def disk_used():
    data = psutil.disk_usage("/").used / (1024 * 1024 * 1024)  # in GB
    system_metrics.append(Metric("DiskUsage", data, "GB", dimension))


def disk_utilization():
    data = psutil.disk_usage("/").percent
    system_metrics.append(Metric("DiskUtilization", data, "percent", dimension))


def disk_available():
    data = psutil.disk_usage("/").free / (1024 * 1024 * 1024)  # in GB
    system_metrics.append(Metric("DiskAvailable", data, "GB", dimension))


def gpu_utilization(num_of_gpu):
    """
    Collect gpu metrics.

    :param num_of_gpu:
    :return:
    """
    if num_of_gpu <= 0:
        return

    # pylint: disable=wrong-import-position
    # pylint: disable=import-outside-toplevel
    # import nvgpu
    # import pynvml
    # from nvgpu import list_gpus

    # pylint: enable=wrong-import-position
    # pylint: enable=import-outside-toplevel

    # info = nvgpu.gpu_info()
    # for value in info:
    #     dimension_gpu = [
    #         Dimension("Level", "Host"),
    #         Dimension("device_id", value["index"]),
    #     ]
    #     system_metrics.append(
    #         Metric(
    #             "GPUMemoryUtilization",
    #             value["mem_used_percent"],
    #             "percent",
    #             dimension_gpu,
    #         )
    #     )
    #     system_metrics.append(
    #         Metric("GPUMemoryUsed", value["mem_used"], "MB", dimension_gpu)
    #     )

    # try:
    #     statuses = list_gpus.device_statuses()
    # except pynvml.nvml.NVMLError_NotSupported:
    #     logging.error("gpu device monitoring not supported")
    #     statuses = []

    # for idx, status in enumerate(statuses):
    #     dimension_gpu = [Dimension("Level", "Host"), Dimension("device_id", idx)]
    #     system_metrics.append(
    #         Metric("GPUUtilization", status["utilization"], "percent", dimension_gpu)
    #     )

    logging.info(f"XPU Utillization: {num_of_gpu}")

    from intel_gpu import list_gpu_info
    info = list_gpu_info(num_of_gpu)
    for line in info[1:]:
        dimension_gpu = [
            Dimension("Level", "Host"),
            Dimension("device_id", line[1]),
        ]
        system_metrics.append(
            Metric("GPUUtilization", line[2], "percent", dimension_gpu)
        )
        system_metrics.append(
            Metric(
                "GPUMemoryUtilization",
                line[3],
                "percent",
                dimension_gpu,
            )
        )
        system_metrics.append(
            Metric("GPUMemoryUsed", line[4], "MB", dimension_gpu)
        )

    # start_time = time.time()
    # timeout = 1
    # try:
    #     # Run the xpu-smi command to get GPU metrics
    #     process = subprocess.Popen(
    #         ["xpu-smi", "dump", "-d", "0", "-m", "0,5"],
    #         stdout=subprocess.PIPE,
    #         text=True  # Ensures that output is in text form
    #     )
    #     output_lines = []
    #     while True:
    #         current_time = time.time()
    #         if current_time - start_time > timeout:
    #             break
    #
    #         # Try to read a line of output
    #         lines = process.stdout.readline()
    #         if not lines:
    #             break
    #         output_lines.append(lines.strip())
    #
    #         # You can process lines here or later
    #         print(lines.strip())  # Example of processing output in real-time
    #
    #     # Parse the output to extract GPU metrics
    #     headers = output_lines[0].split(', ')
    #     data_lines = output_lines[1:]
    #     print(data_lines)
    #     for line in data_lines:
    #         values = line.split(', ')
    #         if len(values) != len(headers):
    #             logging.error(f"Data format error in line: {line}")
    #             continue
    #
    #         # Create a dictionary for easy access to each column
    #         data_dict = dict(zip(headers, values))
    #
    #         # Extract necessary data
    #         timestamp = data_dict["Timestamp"]
    #         device_id = data_dict["DeviceId"]
    #         gpu_utilization = data_dict["GPU Utilization (%)"]
    #         memory_utilization = data_dict["GPU Memory Utilization (%)"]
    #
    #         # Create dimensions
    #         dimensions = [
    #             Dimension("Level", "Host"),
    #             Dimension("DeviceId", device_id)
    #         ]
    #         logging.info(f"created dimension level host, device: {device_id}")
    #         # Append GPU Utilization Metric
    #         system_metrics.append(
    #             Metric(
    #                 "GPUUtilization",
    #                 gpu_utilization,
    #                 "percent",
    #                 dimensions
    #             )
    #         )
    #
    #         # Append GPU Memory Utilization Metric
    #         system_metrics.append(
    #             Metric(
    #                 "GPUMemoryUtilization",
    #                 memory_utilization,
    #                 "percent",
    #                 dimensions
    #             )
    #         )
    #     # logging.info(f"Added metric: {system_metrics[-1]}")
    #     # logging.info(f"Added metric: {system_metrics[-2]}")
    #
    #
    # except FileNotFoundError:
    #     logging.error("xpu-smi command not found. Cannot collect Intel GPU metrics.")
    # except subprocess.CalledProcessError as e:
    #     logging.error("Error running xpu-smi command: %s", e)


def collect_all(mod, num_of_gpu):
    """
    Collect all system metrics.

    :param mod:
    :param num_of_gpu:
    :return:
    """

    members = dir(mod)
    for i in members:
        value = getattr(mod, i)
        if isinstance(value, types.FunctionType) and value.__name__ not in (
            "collect_all",
            "log_msg",
        ):
            if value.__name__ == "gpu_utilization":
                value(num_of_gpu)
            else:
                value()

    for met in system_metrics:
        logging.info(str(met))

    logging.info("")
