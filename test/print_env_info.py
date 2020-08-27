# This script outputs relevant system environment info
# Run it with `python print_env_info.py`.
from __future__ import absolute_import, division, print_function, unicode_literals
import locale
import re
import subprocess
import sys
import os
from collections import namedtuple

try:
    import torch
    TORCH_AVAILABLE = True
except (ImportError, NameError, AttributeError):
    TORCH_AVAILABLE = False

torchserve_env = {
    "torch" : "**Warning torch not present ..",
    "torch_model_archiver" : "N/A",
    "torchserve" : "N/A",
    "torchtext" : "**Warning torchtext not present ..",
    "torchvision" : "**Warning torchvision not present ..",
    "torchaudio" : "**Warning torchaudio not present .."
}

python_env = {
    "python_version" : "N/A",
    "pip_version" : "",
    "pip_packages" : []
}

java_env = {
    "java_version" : []
}

os_info = {
    "os" : "",
    "gcc_version" : "",
    "clang_version" : "N/A",
    "cmake_version" : "N/A"
}

cuda_env = {
    "is_cuda_available" : "No",
    "cuda_runtime_version" : "N/A",
    "nvidia_gpu_models" : [],
    "nvidia_driver_version" : "N/A",
    "cudnn_version" : []
}

def run(command):
    """Returns (return-code, stdout, stderr)"""
    p = subprocess.Popen(command, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, shell=True)
    output, err = p.communicate()
    rc = p.returncode
    enc = locale.getpreferredencoding()
    output = output.decode(enc)
    err = err.decode(enc)
    return rc, output.strip(), err.strip()

def run_and_read_all(run, command):
    """Reads and returns entire output if rc is 0"""
    rc, out, _ = run(command)
    if rc != 0:
        return None
    return out

def run_and_parse_first_match(run, command, regex):
    """Returns the first regex match if it exists"""
    rc, out, _ = run(command)
    if rc != 0:
        return None
    match = re.search(regex, out)
    if match is None:
        return None
    return match.group(1)

def get_pip_packages(run, package_name=None):
    """Returns `pip list` output. """
    # systems generally have `pip` as `pip` or `pip3`
    def run_with_pip(pip):
        if package_name is not None:
            grep_cmd = 'grep "' + package_name + '"'
        else:
            grep_cmd = r'grep "numpy\|pytest\|pylint"'
        return run_and_read_all(run, pip + ' list --format=freeze | ' + grep_cmd)

    # Try to figure out if the user is running pip or pip3.
    out2 = run_with_pip('pip')
    out3 = run_with_pip('pip3')
    num_pips = len([x for x in [out2, out3] if x is not None])
    if num_pips == 0:
        return 'pip', out2
    if num_pips == 1:
        if out2 is not None:
            return 'pip', out2
        return 'pip3', out3
    # num_pips is 2. Return pip3 by default
    return 'pip3', out3

def get_java_version(run):
    rc, out, _ = run("java --version")
    if rc != 0:
        return "**Warning: java not installed..."
    return out

def get_platform():
    if sys.platform.startswith('linux'):
        return 'linux'
    elif sys.platform.startswith('cygwin'):
        return 'cygwin'
    elif sys.platform.startswith('darwin'):
        return 'darwin'
    else:
        return sys.platform

def get_mac_version(run):
    return run_and_parse_first_match(run, 'sw_vers -productVersion', r'(.*)')

def get_lsb_version(run):
    return run_and_parse_first_match(run, 'lsb_release -a', r'Description:\t(.*)')

def check_release_file(run):
    return run_and_parse_first_match(run, 'cat /etc/*-release', r'PRETTY_NAME="(.*)"')

def get_os(run):
    from platform import machine
    platform = get_platform()
    if platform == 'darwin':
        version = get_mac_version(run)
        if version is None:
            return None
        return 'Mac OSX {} ({})'.format(version, machine())
    if platform == 'linux':
        # Ubuntu/Debian based
        desc = get_lsb_version(run)
        if desc is not None:
            return desc
        # Try reading /etc/*-release
        desc = check_release_file(run)
        if desc is not None:
            return desc
        return '{} ({})'.format(platform, machine())
    # Unknown platform
    return platform

def get_gcc_version(run):
    return run_and_parse_first_match(run, 'gcc --version', r'gcc (.*)')

def get_clang_version(run):
    return run_and_parse_first_match(run, 'clang --version', r'clang version (.*)')

def get_cmake_version(run):
    return run_and_parse_first_match(run, 'cmake --version', r'cmake (.*)')

def get_nvidia_driver_version(run):
    if get_platform() == 'darwin':
        cmd = 'kextstat | grep -i cuda'
        return run_and_parse_first_match(run, cmd, r'com[.]nvidia[.]CUDA [(](.*?)[)]')
    smi = get_nvidia_smi()
    return run_and_parse_first_match(run, smi, r'Driver Version: (.*?) ')

def get_gpu_info(run):
    if get_platform() == 'darwin':
        if TORCH_AVAILABLE and torch.cuda.is_available():
            return torch.cuda.get_device_name(None)
        return None
    smi = get_nvidia_smi()
    uuid_regex = re.compile(r' \(UUID: .+?\)')
    rc, out, _ = run(smi + ' -L')
    if rc != 0:
        return None
    # Anonymize GPUs by removing their UUID
    return re.sub(uuid_regex, '', out)

def get_running_cuda_version(run):
    return run_and_parse_first_match(run, 'nvcc --version', r'V(.*)$')

def get_cudnn_version(run):
    """This will return a list of libcudnn.so; it's hard to tell which one is being used"""
    if get_platform() == 'darwin':
        # CUDA libraries and drivers can be found in /usr/local/cuda/. See
        cudnn_cmd = 'ls /usr/local/cuda/lib/libcudnn*'
    else:
        cudnn_cmd = 'ldconfig -p | grep libcudnn | rev | cut -d" " -f1 | rev'
    rc, out, _ = run(cudnn_cmd)
    # find will return 1 if there are permission errors or if not found
    if len(out) == 0 or (rc != 1 and rc != 0):
        l = os.environ.get('CUDNN_LIBRARY')
        if l is not None and os.path.isfile(l):
            return os.path.realpath(l)
        return None
    files = set()
    for fn in out.split('\n'):
        fn = os.path.realpath(fn)  # eliminate symbolic links
        if os.path.isfile(fn):
            files.add(fn)
    if not files:
        return None
    # Alphabetize the result because the order is non-deterministic otherwise
    files = list(sorted(files))
    if len(files) == 1:
        return files[0]
    result = '\n'.join(files)
    return 'Probably one of the following:\n{}'.format(result)

def get_nvidia_smi():
    smi = 'nvidia-smi'
    return smi

def get_torchserve_version():
    try:
        f = open("../ts/version.txt", 'r')
        version = f.readline().rstrip()
    except:
        print("Exception : version.txt file not found !!!")
    finally:
        f.close()
    return version

def get_torch_model_archiver():
    try:
        f = open("../model-archiver/model_archiver/version.txt", 'r')
        version = f.readline().rstrip()
    except:
        print("Exception : version.txt file not found !!!")
    finally:
        f.close()
    return version

def populate_torchserve_env(torch_pkg):
    for pkg in torch_pkg:
        if pkg.split("==")[0] == "torch":
            torchserve_env["torch"] = pkg
        if pkg.split("==")[0] == "torchaudio":
            torchserve_env["torchaudio"] = pkg
        if pkg.split("==")[0] == "torchtext":
            torchserve_env["torchtext"] = pkg
        if pkg.split("==")[0] == "torchvision":
            torchserve_env["torchvision"] = pkg

    torchserve_env["torchserve"] = "torchserve==" + get_torchserve_version()
    torchserve_env["torch_model_archiver"] = "torch-model-archiver==" + get_torch_model_archiver()

def populate_python_env(pip_version, pip_list_output):
    python_env["python_version"] = '{}.{} ({}-bit runtime)'.format(sys.version_info[0], sys.version_info[1], sys.maxsize.bit_length() + 1)
    python_env["pip_version"] = pip_version
    python_env["pip_packages"] = pip_list_output

def populate_java_env():
    java_env["java_version"] = get_java_version(run)

def populate_os_env():
    os_info["os"] = get_os(run)
    os_info["gcc_version"] = get_gcc_version(run)
    os_info["clang_version"] = get_clang_version(run)
    os_info["cmake_version"] = get_cmake_version(run)

def populate_cuda_env(cuda_available_str):    
    cuda_env["is_cuda_available"] = cuda_available_str
    cuda_env["cuda_runtime_version"] = get_running_cuda_version(run)
    cuda_env["nvidia_gpu_models"] = get_gpu_info(run)
    cuda_env["nvidia_driver_version"] = get_nvidia_driver_version(run)
    cuda_env["cudnn_version"] = get_cudnn_version(run)

def populate_env_info():
    #torchserve packages
    _, torch_list_output = get_pip_packages(run,"torch")
    populate_torchserve_env(torch_list_output.split("\n"))

    #python packages
    pip_version, pip_list_output = get_pip_packages(run)
    populate_python_env(pip_version, pip_list_output)
    
    #java environment
    populate_java_env()

    #OS environment
    populate_os_env()

    #cuda environment
    if TORCH_AVAILABLE and torch.cuda.is_available():
        populate_cuda_env("Yes")

env_info_fmt = """
------------------------------------------------------------------------------------------
Environment headers
------------------------------------------------------------------------------------------
Torchserve branch : {torchserve_branch}

{torchserve}
{torch_model_archiver}

Python version: {python_version}

Versions of relevant python libraries:
{pip_packages}
{torch}
{torchtext}
{torchvision}
{torchaudio}

Java Version:
{java_version}

OS: {os}
GCC version: {gcc_version}
Clang version: {clang_version}
CMake version: {cmake_version}
""".strip()

cuda_info_fmt = """
Is CUDA available: {is_cuda_available}
CUDA runtime version: {cuda_runtime_version}
GPU models and configuration: {nvidia_gpu_models}
Nvidia driver version: {nvidia_driver_version}
cuDNN version: {cudnn_version}
"""

def get_pretty_env_info(torchserve_branch):
    global env_info_fmt
    global cuda_info_fmt
    populate_env_info()
    env_dict = {**torchserve_env, **python_env, **java_env, **os_info}
    env_dict["torchserve_branch"] = torchserve_branch

    if TORCH_AVAILABLE and torch.cuda.is_available():
        env_dict.update(cuda_env)
        env_info_fmt = env_info_fmt + "\n" + cuda_info_fmt

    return env_info_fmt.format(**env_dict)

def main(torchserve_branch):
    output = get_pretty_env_info(torchserve_branch)
    print(output)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        torchserve_branch = sys.argv[1]
    else:
        torchserve_branch = "master"
    main(torchserve_branch)