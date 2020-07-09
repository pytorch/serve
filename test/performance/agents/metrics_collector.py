#!/usr/bin/env python3

# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#     http://www.apache.org/licenses/LICENSE-2.0
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

"""
Server metrics collector
"""
# pylint: disable=redefined-builtin, broad-except, unused-variable

import argparse
import logging
import os
import sys
import tempfile
import time
import gevent
import psutil

from utils.process import get_process_pid_from_file, get_child_processes, \
    get_server_processes, get_server_pidfile
from metrics import AVAILABLE_METRICS, get_metrics
import configuration

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, format="%(message)s", level=logging.INFO)

TMP_DIR = tempfile.gettempdir()
METRICS_LOG_FILE = os.path.join(TMP_DIR, "server_metrics_{}.log".format(int(time.time())))
METRICS_COLLECTOR_PID_FILE = os.path.join(TMP_DIR, "metrics_collector.pid")
PID_FILE = configuration.get('server', 'pid_file', 'model_server.pid')

MONITOR_INTERVAL = 1


def store_pid(pid_file):
    """ Store the current process id to pid_file"""
    process = psutil.Process()
    pid_file = os.path.join(pid_file)
    with open(pid_file, "w") as pf:
        pf.write(str(process.pid))


def stop_process(pid_file):
    """This will stop already running process .
       Note at a time only one pid file will be available.
    """
    pid = get_process_pid_from_file(pid_file)
    if pid:
        try:
            process = psutil.Process(pid)
            if process.is_running():
                logger.info("Process with pid %s is running. Killing it.", process.pid)
                process.kill()
        except Exception as e:
            pass
        else:
            logger.info("Dead process with pid %s found in '%s'.", process.pid, pid_file)

        logger.info("Removing pid file '%s'.", pid_file)
        os.remove(pid_file)


def check_is_running(pid_file):
    """check if pid is running"""
    pid = get_process_pid_from_file(pid_file)
    if pid:
        try:
            perf_mon_process = psutil.Process(pid)
        except Exception as e:
            stop_process(pid_file)
        else:
            if perf_mon_process.is_running():
                logger.error("Performance monitoring script already running. "
                             "Stop it using stop option.")
                sys.exit()


def store_metrics_collector_pid():
    """ Store the process id of metrics collector process"""
    store_pid(METRICS_COLLECTOR_PID_FILE)


def stop_metrics_collector_process():
    """This will stop already running metrics collector process.
        Note at a time only one pid file will be available.
     """
    stop_process(METRICS_COLLECTOR_PID_FILE)


def monitor_processes(server_process, metrics, interval, socket):
    """ Monitor the metrics of server_process and its child processes
    """
    while True:
        message = []
        collected_metrics = get_metrics(server_process, get_child_processes(server_process), logger)
        metrics_msg = []
        for metric in metrics:
            message.append(str(collected_metrics.get(metric, 0)))
            if collected_metrics.get(metric) is not None:
                metrics_msg.append("{0} : {1}".format(metric, collected_metrics.get(metric, 0)))

        message = "\t".join(message) + "\t\n"
        logger.info("%s", " -- ".join(metrics_msg))

        if socket:
            try:
                socket.send(message.encode("latin-1"))
            except BrokenPipeError:
                logger.info("Stopping monitoring as socket connection is closed.")
                break

        # TODO - log metrics to a file METRICS_LOG_FILE if METRICS_LOG_FILE is provided
        gevent.sleep(interval)


def start_metric_collection(server_process, metrics, interval, socket):
    bad_metrics = set(metrics) - set(AVAILABLE_METRICS)
    if bad_metrics:
        raise Exception("Metrics not available for monitoring {}.".format(bad_metrics))

    logger.info("Started metric collection for target server processes.....")
    thread = gevent.spawn(monitor_processes, server_process, metrics, interval, socket)
    gevent.joinall([thread])


def start_metric_collector_process():
    """Spawn a metric collection process and keep on monitoring """

    check_is_running(METRICS_COLLECTOR_PID_FILE)
    store_metrics_collector_pid()
    server_pid = get_process_pid_from_file(get_server_pidfile(PID_FILE))
    server_process = get_server_processes(server_pid)
    start_metric_collection(server_process, AVAILABLE_METRICS, MONITOR_INTERVAL, None)


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, format="%(message)s", level=logging.INFO)
    parser = argparse.ArgumentParser(prog='metric-collector', description='System Performance Metrics collector')
    sub_parse = parser.add_mutually_exclusive_group(required=True)
    sub_parse.add_argument('--start', action='store_true', help='Start the metric-collector')
    sub_parse.add_argument('--stop', action='store_true', help='Stop the metric-collector')

    args = parser.parse_args()

    if args.start:
        start_metric_collector_process()
    elif args.stop:
        stop_metrics_collector_process()
