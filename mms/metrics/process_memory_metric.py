

"""
Collect process memory usage metrics  here
Pass a json, collection of pids and gpuID
"""

import logging

import psutil


def get_cpu_usage(pid):
    """
    use psutil for cpu memory
    :param pid: str
    :return: int
    """
    try:
        process = psutil.Process(int(pid))
    except psutil.Error:
        logging.error("Failed get process for pid: %s", pid, exc_info=True)
        return 0

    mem_utilization = process.memory_info()[0]
    return mem_utilization


def check_process_mem_usage(stdin):
    """

    Return
    ------
    mem_utilization: float
    """
    process_list = stdin.readline().strip().split(",")
    for process in process_list:
        if not process:
            continue
        logging.info("%s:%d", process, get_cpu_usage(process))
