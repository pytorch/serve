import os
import platform
import subprocess
import sys

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(REPO_ROOT)

nvidia_smi_cmd = {
    "Windows": "nvidia-smi.exe",
    "Darwin": "nvidia-smi",
    "Linux": "nvidia-smi",
}


def is_gpu_instance():
    return True if os.system(nvidia_smi_cmd[platform.system()]) == 0 else False


def is_conda_build_env():
    return True if os.system("conda-build") == 0 else False


def is_conda_env():
    return True if os.system("conda") == 0 else False


def check_python_version():
    req_version = (3, 8)
    cur_version = sys.version_info

    if not (
        cur_version.major == req_version[0] and cur_version.minor >= req_version[1]
    ):
        print("System version" + str(cur_version))
        print(
            f"TorchServe supports Python {req_version[0]}.{req_version[1]} and higher only. Please upgrade"
        )
        exit(1)


def check_ts_version():
    from ts.version import __version__

    return __version__


def try_and_handle(cmd, dry_run=False):
    if dry_run:
        print(f"Executing command: {cmd}")
    else:
        try:
            subprocess.run([cmd], shell=True, check=True)
        except subprocess.CalledProcessError as e:
            raise (e)
