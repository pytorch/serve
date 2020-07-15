#!/usr/bin/env python3
""" Customised system and Model Server process metrics for monitoring and pass-fail criteria in taurus"""

# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#     http://www.apache.org/licenses/LICENSE-2.0
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.
# pylint: disable=redefined-builtin, redefined-outer-name, broad-except, unused-variable

from enum import Enum
from statistics import mean

import psutil
from psutil import NoSuchProcess, ZombieProcess


class ProcessType(Enum):
    """ Type of Server processes to compute metrics on """
    FRONTEND = 1
    WORKER = 2
    ALL = 3


operators = {
    'sum': sum,
    'avg': mean,
    'min': min,
    'max': max
}

process_metrics = {
    # cpu
    'cpu_percent': lambda p: p.get('cpu_percent', 0),
    'cpu_user_time': lambda p: getattr(p.get('cpu_times', {}), 'user', 0),
    'cpu_system_time': lambda p: getattr(p.get('cpu_times', {}), 'system', 0),
    'cpu_iowait_time': lambda p: getattr(p.get('cpu_times', {}), 'iowait', 0),
    # memory
    'memory_percent': lambda p: p.get('memory_percent', 0),
    'memory_rss': lambda p: getattr(p.get('memory_info', {}), 'rss', 0),
    'memory_vms': lambda p: getattr(p.get('memory_info', {}), 'vms', 0),
    # io
    'io_read_count': lambda p: getattr(p.get('io_counters', {}), 'read_count', 0),
    'io_write_count': lambda p: getattr(p.get('io_counters', {}), 'write_count', 0),
    'io_read_bytes': lambda p: getattr(p.get('io_counters', {}), 'read_bytes', 0),
    'io_write_bytes': lambda p: getattr(p.get('io_counters', {}), 'write_bytes', 0),
    'file_descriptors': lambda p: p.get('num_fds', 0),
    # processes
    'threads': lambda p: p.get('num_threads', 0)
}

system_metrics = {
    'system_disk_used': None,
    'system_memory_percent': None,
    'system_read_count': None,
    'system_write_count': None,
    'system_read_bytes': None,
    'system_write_bytes': None,
}

misc_metrics = {
    'total_processes': None,
    'total_workers': None,
    'orphans': None,
    'zombies': None
}

AVAILABLE_METRICS = list(system_metrics) + list(misc_metrics)
WORKER_NAME = 'model_service_worker.py'

for metric in list(process_metrics):
    for ptype in list(ProcessType):
        if ptype == ProcessType.WORKER:
            PNAME = 'workers'
            for op in list(operators):
                AVAILABLE_METRICS.append('{}_{}_{}'.format(op, PNAME, metric))
        elif ptype == ProcessType.FRONTEND:
            PNAME = 'frontend'
            AVAILABLE_METRICS.append('{}_{}'.format(PNAME, metric))
        else:
            PNAME = 'all'
            for op in list(operators):
                AVAILABLE_METRICS.append('{}_{}_{}'.format(op, PNAME, metric))

children = set()
zombie_children = set()


def get_metrics(server_process, child_processes, logger):
    """ Get Server processes specific metrics
    """
    result = {}
    children.update(child_processes)
    logger.debug("children : {0}".format(",".join([str(c.pid) for c in children])))

    def update_metric(metric_name, proc_type, stats):
        stats = list(filter(lambda x: isinstance(x, (float, int)), stats))
        stats = stats if len(stats) else [0]

        if proc_type == ProcessType.WORKER:
            proc_name = 'workers'
        elif proc_type == ProcessType.FRONTEND:
            proc_name = 'frontend'
            result[proc_name + '_' + metric_name] = stats[0]
            return
        else:
            proc_name = 'all'

        for op_name in operators:
            result['{}_{}_{}'.format(op_name, proc_name, metric_name)] = operators[op_name](stats)

    processes_stats = []
    reclaimed_pids = []

    try:
        # as_dict() gets all stats in one shot
        processes_stats.append({'type': ProcessType.FRONTEND, 'stats': server_process.as_dict()})
    except Exception as e:
        pass

    for child in children | zombie_children:
        try:
            child_cmdline = child.cmdline()
            if psutil.pid_exists(child.pid) and len(child_cmdline) >= 2 and WORKER_NAME in child_cmdline[1]:
                processes_stats.append({'type': ProcessType.WORKER, 'stats': child.as_dict()})
            else:
                reclaimed_pids.append(child)
                logger.debug('child {0} no longer available'.format(child.pid))
        except ZombieProcess:
            zombie_children.add(child)
        except NoSuchProcess:
            reclaimed_pids.append(child)
            logger.debug('child {0} no longer available'.format(child.pid))

    for p in reclaimed_pids:
        if p in children:
            children.remove(p)
        if p in zombie_children:
            zombie_children.remove(p)

    ### PROCESS METRICS ###
    worker_stats = list(map(lambda x: x['stats'], \
                            filter(lambda x: x['type'] == ProcessType.WORKER, processes_stats)))
    server_stats = list(map(lambda x: x['stats'], \
                            filter(lambda x: x['type'] == ProcessType.FRONTEND, processes_stats)))
    all_stats = list(map(lambda x: x['stats'], processes_stats))

    for k in process_metrics:
        update_metric(k, ProcessType.WORKER, list(map(process_metrics[k], worker_stats)))
        update_metric(k, ProcessType.ALL, list(map(process_metrics[k], all_stats)))
        update_metric(k, ProcessType.FRONTEND, list(map(process_metrics[k], server_stats)))

    # Total processes
    result['total_processes'] = len(worker_stats) + 1
    result['total_workers'] = max(len(worker_stats) - 1, 0)
    result['orphans'] = len(list(filter(lambda p: p['ppid'] == 1, worker_stats)))
    result['zombies'] = len(zombie_children)

    # ###SYSTEM METRICS ###
    result['system_disk_used'] = psutil.disk_usage('/').used
    result['system_memory_percent'] = psutil.virtual_memory().percent
    system_disk_io_counters = psutil.disk_io_counters()
    result['system_read_count'] = system_disk_io_counters.read_count
    result['system_write_count'] = system_disk_io_counters.write_count
    result['system_read_bytes'] = system_disk_io_counters.read_bytes
    result['system_write_bytes'] = system_disk_io_counters.write_bytes

    return result


if __name__ == "__main__":
    import logging
    import sys
    from agents.utils.process import *
    from agents import configuration

    logger = logging.getLogger(__name__)
    logging.basicConfig(stream=sys.stdout, format="%(message)s", level=logging.INFO)

    PID_FILE = configuration.get('server', 'pid_file', 'model_server.pid')
    server_pid = get_process_pid_from_file(get_server_pidfile(PID_FILE))
    server_process = get_server_processes(server_pid)
    children = get_child_processes(server_process)

    metrics = get_metrics(server_process, children, logger)

    print(metrics)
