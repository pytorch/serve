import os
from subprocess import Popen


def execute(command, wait=False, stdout=None, stderr=None, shell=True):
    print(command)
    cmd = Popen(
        command,
        shell=shell,
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
