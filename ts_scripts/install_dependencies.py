import argparse
import os
import platform
import shlex
import subprocess
import sys

from print_env_info import run_and_parse_first_match

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(REPO_ROOT)

from ts_scripts.utils import check_python_version

CPP_LINUX_DEPENDENCIES = (
    "autoconf",
    "automake",
    "git",
    "cmake",
    "m4",
    "g++",
    "flex",
    "bison",
    "libgflags-dev",
    "libkrb5-dev",
    "libsasl2-dev",
    "libnuma-dev",
    "pkg-config",
    "libssl-dev",
    "libcap-dev",
    "gperf",
    "libevent-dev",
    "libtool",
    "libboost-all-dev",
    "libjemalloc-dev",
    "libsnappy-dev",
    "wget",
    "unzip",
    "libiberty-dev",
    "liblz4-dev",
    "liblzma-dev",
    "make",
    "zlib1g-dev",
    "binutils-dev",
    "libsodium-dev",
    "libdouble-conversion-dev",
    "ninja-build",
    "clang-tidy",
    "clang-format",
    "build-essential",
    "libgoogle-perftools-dev",
    "rustc",
    "cargo",
    "libunwind-dev",
)

CPP_DARWIN_DEPENDENCIES = (
    "cmake",
    "m4",
    "boost",
    "double-conversion",
    "gperf",
    "libevent",
    "lz4",
    "snappy",
    "xz",
    "openssl",
    "libsodium",
    "icu4c",
    "libomp",
    "llvm",
)

CPP_DARWIN_DEPENDENCIES_LINK = (
    "cmake",
    "boost",
    "double-conversion",
    "gperf",
    "libevent",
    "lz4",
    "snappy",
    "openssl",
    "xz",
    "libsodium",
)


class Common:
    def __init__(self):
        self.torch_stable_url = "https://download.pytorch.org/whl/torch_stable.html"
        self.sudo_cmd = "sudo "

    def check_command(self, command):
        """Check if a command is available on the system."""
        try:
            command = shlex.split(command)
            subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            return True
        except subprocess.CalledProcessError:
            return False
    
    def install_java(self):
        pass

    def install_nodejs(self):
        pass

    def install_torch_packages(self, cuda_version):
        if cuda_version:
            if platform.system() == "Darwin":
                print(
                    "CUDA not supported on MacOS. Refer https://pytorch.org/ for installing from source."
                )
                sys.exit(1)
            elif cuda_version == "cu92" and platform.system() == "Windows":
                print(
                    "CUDA 9.2 not supported on Windows. Refer https://pytorch.org/ for installing from source."
                )
                sys.exit(1)
            else:
                torch_cuda_requirements_file = f"requirements/torch_{cuda_version}_{platform.system().lower()}.txt"
                try:
                    subprocess.run(
                        [sys.executable, "-m", "pip", "install", "-U", "-r", torch_cuda_requirements_file],
                        capture_output=True, text=True, check=True
                    )
                except subprocess.CalledProcessError as e:
                    sys.exit(e.returncode)
        elif args.neuronx:
            torch_neuronx_requirements_file = "requirements/torch_neuronx_linux.txt"
            try:
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "-U", "-r", torch_neuronx_requirements_file],
                    capture_output=True, text=True, check=True
                )
            except subprocess.CalledProcessError as e:
                sys.exit(e.returncode)
        else:
            if platform.machine() == "aarch64":
                requirements_file = f"requirements/torch_{platform.system().lower()}_{platform.machine()}.txt"
            else:
                requirements_file = f"requirements/torch_{platform.system().lower()}.txt"
            try:
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "-U", "-r", requirements_file],
                    capture_output=True, text=True, check=True
                )
            except subprocess.CalledProcessError as e:
                sys.exit(e.returncode)

    def install_python_packages(self, cuda_version, requirements_file_path, nightly):
        check = "where" if platform.system() == "Windows" else "which"
        if subprocess.run([check, "conda"], capture_output=True).returncode == 0:
            # conda install command should run before the pip install commands
            # as it may reinstall the packages with different versions
            print("Conda found. Installing conda-build...")
        try:
            result = subprocess.run(
                ["conda", "install", "-y", "conda-build"],
                capture_output=True, text=True, check=True
            )
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Error installing conda-build: {e.stderr}")
            sys.exit(e.returncode)

        # Install PyTorch packages
        if nightly:
            pt_nightly = "cpu" if not cuda_version else cuda_version
            try:
                subprocess.run(
                    [
                        "pip3", "install", "numpy", "--pre", "torch", "torchvision", "torchaudio", "torchtext",
                        f"--index-url=https://download.pytorch.org/whl/nightly/{pt_nightly}"
                    ],
                    capture_output=True, text=True, check=True
                )
            except subprocess.CalledProcessError as e:
                sys.exit(e.returncode)
        elif args.skip_torch_install:
            print("Skipping Torch installation")
        else:
            self.install_torch_packages(cuda_version)

        # developer.txt also installs packages from common.txt
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-U", "-r", requirements_file_path],
                capture_output=True, text=True, check=True
            )
        except subprocess.CalledProcessError as e:
            sys.exit(e.returncode)

        # Install dependencies for GPU
        if not isinstance(cuda_version, type(None)):
            gpu_requirements_file = os.path.join("requirements", "common_gpu.txt")
            try:
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "-U", "-r", gpu_requirements_file],
                    capture_output=True, text=True, check=True
                )
            except subprocess.CalledProcessError as e:
                sys.exit(e.returncode)

        # Install dependencies for Inferentia2
        if args.neuronx:
            neuronx_requirements_file = os.path.join("requirements", "neuronx.txt")
            try:
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "-U", "-r", neuronx_requirements_file],
                    capture_output=True, text=True, check=True
                )
            except subprocess.CalledProcessError as e:
                sys.exit(e.returncode)

    def install_node_packages(self):
        subprocess.run([{self.sudo_cmd}, "npm", "install", "-g", "newman@5.3.2", "newman-reporter-htmlextra", "markdown-link-check"], check=True)

    def install_jmeter(self):
        pass

    def install_wget(self):
        pass

    def install_numactl(self):
        pass

    def install_cpp_dependencies(self):
        raise NotImplementedError(
            f"Cpp backend is not implemented for platform {platform.system()}"
        )

    def install_neuronx_driver(self):
        pass


class Linux(Common):
    def __init__(self):
        super().__init__()
        # Skip 'sudo ' when the user is root
        self.sudo_cmd = "" if os.geteuid() == 0 else self.sudo_cmd

        if args.force:
            subprocess.run([self.sudo_cmd, "apt-get", "update"], check=True)

    def install_java(self):
        if self.check_command("javac --version") or args.force:
            subprocess.run([self.sudo_cmd, "apt-get", "install", "-y", "openjdk-17-jdk"], check=True)

    def install_nodejs(self):
        if self.check_command("node -v") or args.force:
            subprocess.run([self.sudo_cmd, "curl", "-sL", "https://deb.nodesource.com/setup_18.x", "|", "bash", "-"], check=True)
            subprocess.run([self.sudo_cmd, "apt-get", "install", "-y", "nodejs"], check=True)

    def install_wget(self):
        if self.check_command("wget --version") or args.force:
            subprocess.run([self.sudo_cmd, "apt-get", "install", "-y", "wget"], check=True)

    def install_numactl(self):
        if self.check_command("numactl --show") or args.force:
            subprocess.run([self.sudo_cmd, "apt-get", "install", "-y", "numactl"], check=True)

    def install_cpp_dependencies(self):
        subprocess.run([self.sudo_cmd, ["apt-get", "install", "-y"] + list(CPP_LINUX_DEPENDENCIES)], check=True)

    def install_neuronx_driver(self):
        # Configure Linux for Neuron repository updates
        subprocess.run(["/bin/sh", "-c", ". /etc/os-release && "
                   f"{self.sudo_cmd} tee /etc/apt/sources.list.d/neuron.list > /dev/null <<EOF\n"
                   "deb https://apt.repos.neuron.amazonaws.com ${VERSION_CODENAME} main\n"
                   "EOF\n"], check=True)    
        result = subprocess.run(["wget", "-qO", "-", "https://apt.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB"], check=True, stdout=subprocess.PIPE)
        subprocess.run([self.sudo_cmd, "apt-key", "add", "-"], check=True, input=result.stdout)

        # Update OS packages
        subprocess.run([self.sudo_cmd, "apt-get", "update", "-y"], check=True)

        # Install OS headers
        subprocess.run([self.sudo_cmd, "apt-get", "install", "-y", f"linux-headers-{os.uname().release}"], check=True)

        # install Neuron Driver
        subprocess.run([self.sudo_cmd, "apt-get", "install", "-y", "aws-neuronx-dkms"], check=True)

        # Install Neuron Runtime
        subprocess.run([self.sudo_cmd, "apt-get", "install", "-y", "aws-neuronx-collectives"], check=True)
        subprocess.run([self.sudo_cmd, "apt-get", "install", "-y", "aws-neuronx-runtime-lib"], check=True)

class Windows(Common):
    def __init__(self):
        super().__init__()
        self.sudo_cmd = ""

    def install_java(self):
        pass

    def install_nodejs(self):
        pass

    def install_wget(self):
        pass

    def install_numactl(self):
        pass

    def install_neuronx_driver(self):
        pass


class Darwin(Common):
    def __init__(self):
        super().__init__()

    def install_java(self):
        if self.check_command("javac -version") or args.force:
            out = get_brew_version()
            if out == "N/A":
                sys.exit("**Error: Homebrew not installed...")
            subprocess.run(["brew", "install", "openjdk@17"], check=True)

    def install_nodejs(self):
        subprocess.run(["brew", "unlink", "node"], check=True)
        subprocess.run(["brew", "install", "node@18"], check=True)
        subprocess.run(["brew", "link", "--overwrite", "node@18"], check=True)

    def install_node_packages(self):
        subprocess.run([self.sudo_cmd, "./ts_scripts/mac_npm_deps"], check=True)

    def install_wget(self):
        if self.check_command("wget --version") or args.force:
            subprocess.run(["brew", "install", "wget"], check=True)

    def install_numactl(self):
        if self.check_command("numactl --show") or args.force:
            subprocess.run(["brew", "install", "numactl"], check=True)

    def install_cpp_dependencies(self):
        if self.check_command("clang-tidy --version") or args.force:
            subprocess.run(["brew", "install", "-f"] + list(CPP_DARWIN_DEPENDENCIES), check=True)
            subprocess.run(["brew", "link"] + list(CPP_DARWIN_DEPENDENCIES_LINK), check=True)
            subprocess.run([
                self.sudo_cmd, "ln", "-s", f"{self._get_brew_prefix('llvm')}/bin/clang-format", "/usr/local/bin/clang-format"
            ], check=True)
            subprocess.run([
                self.sudo_cmd, "ln", "-s", f"{self._get_brew_prefix('llvm')}/bin/clang-tidy", "/usr/local/bin/clang-tidy"
            ], check=True)
            subprocess.run([
                self.sudo_cmd, "ln", "-s", f"{self._get_brew_prefix('llvm')}/bin/clang-apply-replacements", "/usr/local/bin/clang-apply-replacements"
            ], check=True)

    def install_neuronx_driver(self):
        pass


def install_dependencies(cuda_version=None, nightly=False):
    os_map = {"Linux": Linux, "Windows": Windows, "Darwin": Darwin}
    system = os_map[platform.system()]()

    if args.environment == "dev":
        system.install_wget()
        system.install_nodejs()
        system.install_node_packages()
        system.install_numactl()

    # Sequence of installation to be maintained
    system.install_java()

    if args.neuronx:
        system.install_neuronx_driver()

    requirements_file = "common.txt" if args.environment == "prod" else "developer.txt"
    requirements_file_path = os.path.join("requirements", requirements_file)

    system.install_python_packages(cuda_version, requirements_file_path, nightly)

    if args.cpp:
        system.install_cpp_dependencies()


def get_brew_version():
    """Returns `brew --version` output."""

    return run_and_parse_first_match("brew --version", r"Homebrew (.*)")


if __name__ == "__main__":
    check_python_version()
    parser = argparse.ArgumentParser(
        description="Install various build and test dependencies of TorchServe"
    )
    parser.add_argument(
        "--cuda",
        default=None,
        choices=[
            "cu92",
            "cu101",
            "cu102",
            "cu111",
            "cu113",
            "cu116",
            "cu117",
            "cu118",
            "cu121",
        ],
        help="CUDA version for torch",
    )
    parser.add_argument(
        "--neuronx",
        action="store_true",
        help="Install dependencies for inferentia2 support",
    )
    parser.add_argument(
        "--cpp",
        action="store_true",
        help="Install dependencies for cpp backend",
    )
    parser.add_argument(
        "--environment",
        default="prod",
        choices=["prod", "dev"],
        help="environment(production or developer) on which dependencies will be installed",
    )

    parser.add_argument(
        "--nightly_torch",
        action="store_true",
        help="Install nightly version of torch package",
    )

    parser.add_argument(
        "--skip_torch_install",
        action="store_true",
        help="Skip Torch installation",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="force reinstall dependencies wget, node, java and apt-update",
    )
    args = parser.parse_args()

    install_dependencies(cuda_version=args.cuda, nightly=args.nightly_torch)
