import argparse
import logging
import sys
import types
from builtins import str

from intel_gpu import list_gpu_info

from ts.metrics.dimension import Dimension
from ts.metrics.metric import Metric
from ts.metrics.process_memory_metric import check_process_mem_usage

intel_gpu_system_metrics = []
dimension = [Dimension("Level", "Host")]


def gpu_utilization(num_of_gpu):
    """
    Collect gpu metrics.

    :param num_of_gpu:
    :return:
    """
    if num_of_gpu <= 0:
        return

    info = list_gpu_info(num_of_gpu)
    for line in info[1:]:
        dimension_gpu = [
            Dimension("Level", "Host"),
            Dimension("device_id", int(line[1])),
        ]
        if line[2] != "N/A":
            intel_gpu_system_metrics.append(
                Metric("GPUUtilization", float(line[2]), "percent", dimension_gpu)
            )
        if line[3] != "N/A":
            intel_gpu_system_metrics.append(
                Metric(
                    "GPUMemoryUtilization",
                    float(line[3]),
                    "percent",
                    dimension_gpu,
                )
            )
        if line[4] != "N/A":
            intel_gpu_system_metrics.append(
                Metric("GPUMemoryUsed", float(line[4]), "MB", dimension_gpu)
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
                gpu_utilization(num_of_gpu)
            else:
                value()

    for met in mod.system_metrics:
        logging.info(str(met))

    for met in intel_gpu_system_metrics:
        logging.info(str(met))

    logging.info("")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--gpu", action="store", help="number of GPU", type=int)
    arguments = parser.parse_args()

    logging.basicConfig(stream=sys.stdout, format="%(message)s", level=logging.INFO)

    collect_all(sys.modules["ts.metrics.system_metrics"], arguments.gpu)

    check_process_mem_usage(sys.stdin)
