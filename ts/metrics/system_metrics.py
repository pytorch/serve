"""
Module to collect system metrics for front-end
"""

import logging
import types
from builtins import str

import psutil
import torch

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


def collect_gpu_metrics(num_of_gpus):
    """
    Collect GPU metrics. Supports NVIDIA and AMD GPUs.
    :param num_of_gpus: Total number of available GPUs.
    :return:
    """
    if num_of_gpus <= 0:
        return
    for gpu_index in range(num_of_gpus):
        if torch.version.cuda:
            free, total = torch.cuda.mem_get_info(gpu_index)
            mem_used = (total - free) // 1024**2
            gpu_mem_utilization = torch.cuda.memory_usage(gpu_index)
            gpu_utilization = torch.cuda.utilization(gpu_index)
        elif torch.version.hip:
            # There is currently a bug in
            # https://github.com/pytorch/pytorch/blob/838958de94ed3b9021ddb395fe3e7ed22a60b06c/torch/cuda/__init__.py#L1171
            # which does not capture the rate/percentage correctly.
            # Otherwise same methods could be used.
            # https://rocm.docs.amd.com/projects/amdsmi/en/latest/how-to/using-amdsmi-for-python.html#amdsmi-get-gpu-activity
            import amdsmi

            try:
                amdsmi.amdsmi_init()

                handle = amdsmi.amdsmi_get_processor_handles()[gpu_index]
                mem_used = amdsmi.amdsmi_get_gpu_vram_usage(handle)["vram_used"]
                engine_usage = amdsmi.amdsmi_get_gpu_activity(handle)
                gpu_utilization = engine_usage["gfx_activity"]
                gpu_mem_utilization = engine_usage["umc_activity"]
            except amdsmi.AmdSmiException as e:
                logging.error("Could not initialize AMD-SMI library.")
            finally:
                try:
                    amdsmi.amdsmi_shut_down()
                except amdsmi.AmdSmiException as e:
                    logging.error("Could not shut down AMD-SMI library.")
        elif torch.backends.mps.is_available():
            try:
                total_memory = torch.mps.driver_allocated_memory()
                mem_used = torch.mps.current_allocated_memory()
                gpu_mem_utilization = (
                    (mem_used / total_memory * 100) if total_memory > 0 else 0
                )
                # Currently there is no way to calculate GPU utilization with MPS.
                gpu_utilization = None
            except Exception as e:
                logging.error(f"Could not capture MPS memory metrics")
                mem_used = 0
                gpu_mem_utilization = 0
                gpu_utilization = None

        dimension_gpu = [
            Dimension("Level", "Host"),
            Dimension("device_id", gpu_index),
        ]
        system_metrics.append(
            Metric(
                "GPUMemoryUtilization",
                gpu_mem_utilization,
                "percent",
                dimension_gpu,
            )
        )
        system_metrics.append(Metric("GPUMemoryUsed", mem_used, "MB", dimension_gpu))
        system_metrics.append(
            Metric("GPUUtilization", gpu_utilization, "percent", dimension_gpu)
        )


def collect_all(mod, num_of_gpus):
    """
    Collect all system metrics.

    :param mod:
    :param num_of_gpus: Total number of available GPUs.
    :return:
    """
    members = dir(mod)
    for i in members:
        value = getattr(mod, i)
        if isinstance(value, types.FunctionType) and value.__name__ not in (
            "collect_all",
        ):
            if value.__name__ == "collect_gpu_metrics":
                value(num_of_gpus)
            else:
                value()

    for met in system_metrics:
        logging.info(str(met))

    logging.info("")
