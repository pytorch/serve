import os
import platform
import sys

nvidia_smi_cmd = {'Windows': 'nvidia-smi.exe',
                         'Darwin': 'nvidia-smi',
                         'Linux': 'nvidia-smi'}

def is_gpu_instance():
    return True if os.system(nvidia_smi_cmd[platform.system()]) == 0 else False

def is_conda_build_env():
    return True if os.system("conda-build") == 0 else False

def is_conda_env():
    return True if os.system("conda") == 0 else False


def check_python_version():
    req_version = (3, 6)
    cur_version = sys.version_info

    if not (cur_version.major == req_version[0] and cur_version.minor >= req_version[1]):
        print("System version" + str(cur_version))
        print("TorchServe supports Python 3.6 and higher only. Please upgrade")
        exit(1)