import os
import shlex
from subprocess import Popen


def execute(command, wait=False, stdout=None, stderr=None):
    print(command)
    # Split the command into a list of arguments
    if isinstance(command, str):
        command = shlex.split(command)

    cmd = Popen(
        command,
        close_fds=True,
        stdout=stdout,
        stderr=stderr,
        universal_newlines=True,
    )
    if wait:
        cmd.wait()
    return cmd


def is_workflow(model_url):
    return model_url.endswith(".war")


def is_file_empty(file_path):
    """Check if file is empty by confirming if its size is 0 bytes"""
    # Check if file exist and it is empty
    return os.path.exists(file_path) and os.stat(file_path).st_size == 0
