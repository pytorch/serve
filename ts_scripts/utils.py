import os
import platform
import shlex
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
    cmd = nvidia_smi_cmd.get(platform.system())

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        return True
    else:
        return False


def is_conda_build_env():
    result = subprocess.run(["conda-build"], capture_output=True, text=True)
    if result.returncode == 0:
        return True
    else:
        return False


def is_conda_env():
    result = subprocess.run(["conda"], capture_output=True, text=True)
    if result.returncode == 0:
        return True
    else:
        return False


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
    if isinstance(cmd, str):
        cmd = shlex.split(cmd)
    else:
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            raise (e)

def find_conda_binary():
    try:
        subprocess.run(["conda", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        conda_path = subprocess.check_output(["which", "conda"], text=True).strip()
    except subprocess.CalledProcessError:
        conda_path = os.path.expanduser("$HOME/miniconda/condabin/conda")
    return conda_path
