# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#     http://www.apache.org/licenses/LICENSE-2.0
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

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
