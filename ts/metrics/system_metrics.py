"""
Module to collect system metrics for front-end
"""
import logging
import types
from builtins import str
import os
import psutil
import subprocess

from ts.metrics.dimension import Dimension
from ts.metrics.metric import Metric

system_metrics = []
dimension = [Dimension('Level', 'Host')]


def cpu_utilization():
    data = psutil.cpu_percent()
    system_metrics.append(Metric('CPUUtilization', data, 'percent', dimension))


def memory_used():
    data = psutil.virtual_memory().used / (1024 * 1024)  # in MB
    system_metrics.append(Metric('MemoryUsed', data, 'MB', dimension))


def memory_available():
    data = psutil.virtual_memory().available / (1024 * 1024)  # in MB
    system_metrics.append(Metric('MemoryAvailable', data, 'MB', dimension))


def memory_utilization():
    data = psutil.virtual_memory().percent
    system_metrics.append(Metric('MemoryUtilization', data, 'percent', dimension))


def disk_used():
    data = psutil.disk_usage('/').used / (1024 * 1024 * 1024)  # in GB
    system_metrics.append(Metric('DiskUsage', data, 'GB', dimension))


def disk_utilization():
    data = psutil.disk_usage('/').percent
    system_metrics.append(Metric('DiskUtilization', data, 'percent', dimension))


def disk_available():
    data = psutil.disk_usage('/').free / (1024 * 1024 * 1024)  # in GB
    system_metrics.append(Metric('DiskAvailable', data, 'GB', dimension))

def gpu_utilization():
    if num_of_gpu <= 0:
        return
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=utilization.gpu,utilization.memory", "--format=csv,nounits,noheader", ],
        encoding="utf-8",
        capture_output=True,  # valid for python version >=3.7
        check=True,
    )
    for idx, values in enumerate(result.stdout.strip().split(os.linesep)):
        gpu_memory = values.split(", ")
        dimension_gpu = [Dimension('Level', 'Host'), Dimension("device_id", idx)]
        system_metrics.append(Metric('GPUUtilization', gpu_memory[0], 'percent', dimension_gpu))
        system_metrics.append(Metric('GPUMemoryUtilization', gpu_memory[1], 'percent', dimension_gpu))

num_of_gpu = -1
def collect_all(mod, gpu):
    """
    Collect all system metrics.

    :param mod:
    :return:
    """
    global num_of_gpu = gpu
    members = dir(mod)
    for i in members:
        value = getattr(mod, i)
        if isinstance(value, types.FunctionType) and value.__name__ not in ('collect_all', 'log_msg'):
            value()

    for met in system_metrics:
        logging.info(str(met))

    logging.info("")
