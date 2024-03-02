# This script outputs relevant system environment info
# Run it with `python print_env_info.py`.
import locale
import os
import re
import subprocess
import sys

try:
    import torch

    TORCH_AVAILABLE = True
except (ImportError, NameError, AttributeError):
    TORCH_AVAILABLE = False

torchserve_env = {
    "torch": "**Warning: torch not present ..",
    "torch_model_archiver": "**Warning: torch-model-archiver not installed ..",
    "torchserve": "**Warning: torchserve not installed ..",
    "torchtext": "**Warning: torchtext not present ..",
    "torchvision": "**Warning: torchvision not present ..",
    "torchaudio": "**Warning: torchaudio not present ..",
}

python_env = {
    "python_version": "N/A",
    "python_executable_path": "N/A",
    "pip_version": "",
    "pip_packages": [],
}

java_env = {"java_version": []}

os_info = {"os": "", "gcc_version": "", "clang_version": "N/A", "cmake_version": "N/A"}

cuda_env = {
    "is_cuda_available": "No",
    "cuda_runtime_version": "N/A",
    "nvidia_gpu_models": [],
    "nvidia_driver_version": "N/A",
    "cudnn_version": [],
}

npm_env = {"npm_pkg_version": []}

cpp_env = {"LIBRARY_PATH": ""}


def get_nvidia_smi():
    # Note: nvidia-smi is currently available only on Windows and Linux
    smi = "nvidia-smi"
    if get_platform() == "win32":
        smi = "nvidia-smi.exe"
    return smi


def run(command):
    """Returns (return-code, stdout, stderr)"""
    p = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
    )
    output, err = p.communicate()
    rc = p.returncode
    enc = locale.getpreferredencoding()
    output = output.decode(enc)
    err = err.decode(enc)
    if output.startswith("├── "):
        return rc, output.strip().replace("├── ", ""), err.strip()
    return rc, output.strip(), err.strip()


def run_and_read_all(command):
    """Reads and returns entire output if rc is 0"""
    rc, out, _ = run(command)
    if rc != 0:
        return "N/A"
    return out


def run_and_parse_first_match(command, regex):
    """Returns the first regex match if it exists"""
    rc, out, _ = run(command)
    if rc != 0:
        return "N/A"
    match = re.search(regex, out)
    if match is None:
        return "N/A"
    return match.group(1)


def get_npm_packages():
    """Returns `npm ls -g --depth=0` output."""

    grep_cmd = r'grep "newman\|markdown-link-check"'
    out = run_and_read_all("npm ls -g --depth=0 | " + grep_cmd)
    if out == "N/A":
        return "**Warning: newman, newman-reporter-html markdown-link-check not installed..."
    return out


def get_pip_packages(package_name=None):
    """Returns `pip list` output."""

    # systems generally have `pip` as `pip` or `pip3`
    def run_with_pip(pip):
        if get_platform() == "win32":
            system_root = os.environ.get("SYSTEMROOT", "C:\\Windows")
            findstr_cmd = os.path.join(system_root, "System32", "findstr")
            grep_cmd = r'{} /R "numpy torch"'.format(findstr_cmd)
        elif package_name == "torch":
            grep_cmd = 'grep "' + package_name + '"'
        else:
            grep_cmd = r'grep "numpy\|pytest\|pylint\|transformers\|psutil\|wheel\|requests\|sentencepiece\|pillow\|captum\|nvgpu\|pygit2\|torch"'
        return run_and_read_all(pip + " list --format=freeze | " + grep_cmd)

    out = run_with_pip("pip3")
    if out == "N/A":
        out = None
    return "pip3", out


def get_java_version():
    rc, out, _ = run("java -version")
    if rc != 0:
        return "**Warning: java not installed..."
    return out


def get_platform():
    if sys.platform.startswith("linux"):
        return "linux"
    elif sys.platform.startswith("win32"):
        return "win32"
    elif sys.platform.startswith("cygwin"):
        return "cygwin"
    elif sys.platform.startswith("darwin"):
        return "darwin"
    else:
        return sys.platform


def get_mac_version():
    return run_and_parse_first_match("sw_vers -productVersion", r"(.*)")


def get_lsb_version():
    return run_and_parse_first_match("lsb_release -a", r"Description:\t(.*)")


def get_windows_version():
    system_root = os.environ.get("SYSTEMROOT", "C:\\Windows")
    wmic_cmd = os.path.join(system_root, "System32", "Wbem", "wmic")
    findstr_cmd = os.path.join(system_root, "System32", "findstr")
    return run_and_read_all(
        "{} os get Caption | {} /v Caption".format(wmic_cmd, findstr_cmd)
    )


def check_release_file():
    return run_and_parse_first_match("cat /etc/*-release", r'PRETTY_NAME="(.*)"')


def get_os():
    from platform import machine

    platform = get_platform()
    if platform == "win32" or platform == "cygwin":
        return get_windows_version()
    if platform == "darwin":
        version = get_mac_version()
        if version is None:
            return None
        return "Mac OSX {} ({})".format(version, machine())
    if platform == "linux":
        # Ubuntu/Debian based
        desc = get_lsb_version()
        if desc is not None:
            return desc
        # Try reading /etc/*-release
        desc = check_release_file()
        if desc is not None:
            return desc
        return "{} ({})".format(platform, machine())
    # Unknown platform
    return platform


def get_gcc_version():
    return run_and_parse_first_match("gcc --version", r"gcc (.*)")


def get_clang_version():
    return run_and_parse_first_match("clang --version", r"clang version (.*)")


def get_cmake_version():
    return run_and_parse_first_match("cmake --version", r"cmake (.*)")


def get_nvidia_driver_version():
    smi = get_nvidia_smi()
    if get_platform() == "darwin":
        cmd = "kextstat | grep -i cuda"
        return run_and_parse_first_match(cmd, r"com[.]nvidia[.]CUDA [(](.*?)[)]")

    return run_and_parse_first_match(smi, r"Driver Version: (.*?) ")


def get_nvidia_gpu_info():
    smi = get_nvidia_smi()
    if get_platform() == "darwin":
        if TORCH_AVAILABLE and torch.cuda.is_available():
            return torch.cuda.get_device_name(None)
        return None
    uuid_regex = re.compile(r" \(UUID: .+?\)")
    rc, out, _ = run(smi + " -L")
    if rc != 0:
        return None
    # Anonymize GPUs by removing their UUID
    return "\n" + re.sub(uuid_regex, "", out)


def get_running_cuda_version():
    return run_and_parse_first_match("nvcc --version", r"V([\d.]+)")


def get_cudnn_version():
    """This will return a list of libcudnn.so; it's hard to tell which one is being used"""
    if get_platform() == "win32":
        system_root = os.environ.get("SYSTEMROOT", "C:\\Windows")
        cuda_path = os.environ.get("CUDA_PATH", "%CUDA_PATH%")
        where_cmd = os.path.join(system_root, "System32", "where")
        cudnn_cmd = '{} /R "{}\\bin" cudnn*.dll'.format(where_cmd, cuda_path)
    elif get_platform() == "darwin":
        # CUDA libraries and drivers can be found in /usr/local/cuda/. See
        cudnn_cmd = "ls /usr/local/cuda/lib/libcudnn*"
    else:
        cudnn_cmd = 'ldconfig -p | grep libcudnn | rev | cut -d" " -f1 | rev'
    rc, out, _ = run(cudnn_cmd)
    # find will return 1 if there are permission errors or if not found
    if len(out) == 0 or (rc != 1 and rc != 0):
        l = os.environ.get("CUDNN_LIBRARY")
        if l is not None and os.path.isfile(l):
            return os.path.realpath(l)
        return None
    files = set()
    for fn in out.split("\n"):
        fn = os.path.realpath(fn)  # eliminate symbolic links
        if os.path.isfile(fn):
            files.add(fn)
    if not files:
        return None
    # Alphabetize the result because the order is non-deterministic otherwise
    files = list(sorted(files))
    if len(files) == 1:
        return files[0]
    result = "\n".join(files)
    return "Probably one of the following:\n{}".format(result)


def get_torchserve_version():
    # fetch the torchserve version from version.txt file
    with open(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "ts", "version.txt"
        ),
        "r",
    ) as file:
        version = file.readline().rstrip()
    return version


def get_torch_model_archiver():
    # fetch the torch-model-archiver version from version.txt file
    with open(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "model-archiver",
            "model_archiver",
            "version.txt",
        ),
        "r",
    ) as file:
        version = file.readline().rstrip()
    return version


def get_library_path():
    platform = get_platform()
    if platform == "darwin":
        return os.environ.get("DYLD_LIBRARY_PATH", "")
    elif platform == "linux":
        return os.environ.get("LD_LIBRARY_PATH", "")
    else:
        return ""


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
        if pkg.split("==")[0] == "torchserve" and len(torchserve_branch) == 0:
            torchserve_env["torchserve"] = pkg
        if pkg.split("==")[0] == "torch-model-archiver" and len(torchserve_branch) == 0:
            torchserve_env["torch_model_archiver"] = pkg
    if len(torchserve_branch) > 0:
        torchserve_env["torchserve"] = "torchserve==" + get_torchserve_version()
        torchserve_env["torch_model_archiver"] = (
            "torch-model-archiver==" + get_torch_model_archiver()
        )


def populate_python_env(pip_version, pip_list_output):
    python_env["python_version"] = (
        f"{sys.version_info[0]}.{sys.version_info[1]} "
        f"({sys.maxsize.bit_length() + 1}-bit runtime)"
    )
    python_env["python_executable_path"] = sys.executable
    python_env["pip_version"] = pip_version
    python_env["pip_packages"] = pip_list_output


def populate_java_env():
    java_env["java_version"] = get_java_version()


def populate_os_env():
    os_info["os"] = get_os()
    os_info["gcc_version"] = get_gcc_version()
    os_info["clang_version"] = get_clang_version()
    os_info["cmake_version"] = get_cmake_version()


def populate_cuda_env(cuda_available_str):
    cuda_env["is_cuda_available"] = cuda_available_str
    cuda_env["cuda_runtime_version"] = get_running_cuda_version()
    cuda_env["nvidia_gpu_models"] = get_nvidia_gpu_info()
    cuda_env["nvidia_driver_version"] = get_nvidia_driver_version()
    cuda_env["cudnn_version"] = get_cudnn_version()


def populate_npm_env():
    npm_env["npm_pkg_version"] = get_npm_packages()


def populate_cpp_env():
    cpp_env["LIBRARY_PATH"] = get_library_path()


def populate_env_info():
    # torchserve packages
    _, torch_list_output = get_pip_packages("torch")
    if torch_list_output is not None:
        populate_torchserve_env(torch_list_output.split("\n"))

    # python packages
    pip_version, pip_list_output = get_pip_packages()
    populate_python_env(pip_version, pip_list_output)

    # java environment
    populate_java_env()

    # OS environment
    populate_os_env()

    # cuda environment
    if TORCH_AVAILABLE and torch.cuda.is_available():
        populate_cuda_env("Yes")

    if get_platform() == "darwin":
        populate_npm_env()

    if get_platform() in ("darwin", "linux"):
        populate_cpp_env()


env_info_fmt = """
------------------------------------------------------------------------------------------
Environment headers
------------------------------------------------------------------------------------------
Torchserve branch: {torchserve_branch}

{torchserve}
{torch_model_archiver}

Python version: {python_version}
Python executable: {python_executable_path}

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

npm_info_fmt = """
Versions of npm installed packages:
{npm_pkg_version}
"""

cpp_env_info_fmt = """
Environment:
library_path (LD_/DYLD_): {LIBRARY_PATH}
"""


def get_pretty_env_info(branch_name):
    global env_info_fmt
    global cuda_info_fmt
    global npm_info_fmt
    global cpp_env_info_fmt
    populate_env_info()
    env_dict = {
        **torchserve_env,
        **python_env,
        **java_env,
        **os_info,
        "torchserve_branch": branch_name,
        **cpp_env,
    }

    if TORCH_AVAILABLE and torch.cuda.is_available():
        env_dict.update(cuda_env)
        env_info_fmt = env_info_fmt + "\n" + cuda_info_fmt

    if get_platform() == "darwin":
        env_dict.update(npm_env)
        env_info_fmt = env_info_fmt + "\n" + npm_info_fmt

    if get_platform() in ("darwin", "linux"):
        env_info_fmt = env_info_fmt + "\n" + cpp_env_info_fmt

    return env_info_fmt.format(**env_dict)


def main(branch_name):
    global torchserve_branch
    torchserve_branch = branch_name
    output = get_pretty_env_info(branch_name)
    print(output)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        torchserve_branch = sys.argv[1]
    else:
        torchserve_branch = ""
    main(torchserve_branch)
