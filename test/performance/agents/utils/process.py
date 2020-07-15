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
Utility methods for process related information
"""
# pylint: disable=redefined-builtin

import os
import tempfile
import psutil


def find_procs_by_name(name):
    """Return a list of processes matching 'name'."""
    ls = []
    for p in psutil.process_iter(["name", "exe", "cmdline"]):
        if name == p.info['name'] or \
                p.info['exe'] and os.path.basename(p.info['exe']) == name or \
                p.info['cmdline'] and p.info['cmdline'][0] == name:
            ls.append(p)

    if len(ls) > 1:
        raise Exception("Multiple processes found with name {}.".format(name))

    return ls[0]


def get_process_pid_from_file(file_path):
    """Get the process pid from pid file.
    """
    pid = None
    if os.path.isfile(file_path):
        with open(file_path, "r") as f:
            pid = int(f.readline())

    return pid


def get_child_processes(process):
    """Get all running child processes recursively"""
    child_processes = set()
    for p in process.children(recursive=True):
        child_processes.add(p)
    return child_processes


def get_server_processes(server_process_pid):
    """get psutil Process object from process id """
    try:
        server_process = psutil.Process(server_process_pid)
    except Exception as e:
        print("Server process not found. Error: {}".format(str(e)))
        raise
    return server_process


def get_server_pidfile(file):
    """get temp server pid file"""
    return os.path.join(tempfile.gettempdir(), ".{}".format(file))
