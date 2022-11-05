"""
Module to collect system metrics for front-end
"""
import logging
import types
from builtins import str

import psutil

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
    import nvgpu
    import pynvml
    from nvgpu import list_gpus

    # pylint: enable=wrong-import-position
    # pylint: enable=import-outside-toplevel

    info = nvgpu.gpu_info()
    for value in info:
        dimension_gpu = [
            Dimension("Level", "Host"),
            Dimension("device_id", value["index"]),
        ]
        system_metrics.append(
            Metric(
                "GPUMemoryUtilization",
                value["mem_used_percent"],
                "percent",
                dimension_gpu,
            )
        )
        system_metrics.append(
            Metric("GPUMemoryUsed", value["mem_used"], "MB", dimension_gpu)
        )

    try:
        statuses = list_gpus.device_statuses()
    except pynvml.nvml.NVMLError_NotSupported:
        logging.error("gpu device monitoring not supported")
        statuses = []

    for idx, status in enumerate(statuses):
        dimension_gpu = [Dimension("Level", "Host"), Dimension("device_id", idx)]
        system_metrics.append(
            Metric("GPUUtilization", status["utilization"], "percent", dimension_gpu)
        )


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
