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


def collect_all(mod):
    """
    Collect all system metrics.

    :param mod:
    :return:
    """
    members = dir(mod)
    for i in members:
        value = getattr(mod, i)
        if isinstance(value, types.FunctionType) and value.__name__ not in ('collect_all', 'log_msg'):
            value()

    for met in system_metrics:
        logging.info(str(met))

    logging.info("")
