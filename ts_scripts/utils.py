import os
import sys


def is_gpu_instance():
    return True if os.system("nvidia-smi") == 0 else False


def is_conda_env():
    return True if os.system("conda") == 0 else False


def check_python_version():
    py_version = sys.version
    if not py_version.startswith('3.'):
        print("TorchServe supports Python 3.x only. Please upgrade")
        exit(1)