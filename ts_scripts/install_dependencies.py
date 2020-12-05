import os
import platform
import argparse
import sys


REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(REPO_ROOT)

from ts_scripts.utils import check_python_version, is_gpu_instance


class Common():
    def __init__(self):
        # Assumption is nvidia-smi is installed on systems with gpu
        self.is_gpu_instance = is_gpu_instance()
        self.torch_stable_url = "https://download.pytorch.org/whl/torch_stable.html"
        self.sudo_cmd = 'sudo '

    def install_java(self):
        pass

    def install_nodejs(self):
        pass

    def install_torch_packages(self, cuda_version):
        if self.is_gpu_instance and cuda_version:
            os.system(f"pip install -U -r requirements/torch_{cuda_version}.txt -f {self.torch_stable_url}")
        else:
            os.system(f"pip install -U -r requirements/torch.txt")

    def install_python_packages(self, cuda_version):
        self.install_torch_packages(cuda_version)
        os.system("pip install -U pip setuptools")
        # developer.txt also installs packages from common.txt
        os.system("pip install -U -r requirements/developer.txt")
        # If conda is available install conda-build package
        if os.system("conda") == 0:
            os.system("conda install -y conda-build")

    def install_node_packages(self):
        os.system(f"{self.sudo_cmd}npm install -g newman newman-reporter-html markdown-link-check")

    def install_jmeter(self):
        pass


class Linux(Common):
    def __init__(self):
        super().__init__()
        os.system(f"{self.sudo_cmd}apt-get update")

    def install_java(self):
        os.system(f"{self.sudo_cmd}apt-get install -y openjdk-11-jdk")

    def install_nodejs(self):
        os.system(f"{self.sudo_cmd}curl -sL https://deb.nodesource.com/setup_14.x | {self.sudo_cmd}bash -")
        os.system(f"{self.sudo_cmd}apt-get install -y nodejs")
        os.system(f"{self.sudo_cmd}ln -sf /usr/bin/python3 /usr/bin/python")
        os.system(f"{self.sudo_cmd}ln -sf /usr/bin/pip3 /usr/bin/pip")


class Windows(Common):
    def __init__(self):
        super().__init__()
        self.sudo_cmd = ''

    def install_java(self):
        pass

    def install_nodejs(self):
        pass


class Darwin(Common):
    def __init__(self):
        super().__init__()

    def install_java(self):
        os.system("brew tap AdoptOpenJDK/openjdk")
        os.system("brew cask install adoptopenjdk11")

    def install_nodejs(self):
        os.system("brew unlink node")
        os.system("brew install node@14")
        os.system("brew link --overwrite node@14")

    def install_torch_packages(self, cuda_version=''):
        os.system(f"pip install -U -r requirements/torch.txt -f {self.torch_stable_url}")


def install_dependencies(cuda_version=None):
    os_map = {
        "Linux": Linux,
        "Windows": Windows,
        "Darwin": Darwin
    }
    system = os_map[platform.system()]()

    # Sequence of installation to be maintained
    system.install_java()
    system.install_nodejs()
    system.install_python_packages(cuda_version)
    system.install_node_packages()


if __name__ == "__main__":
    check_python_version()
    parser = argparse.ArgumentParser(description="Install various build and test dependencies of TorchServe")
    parser.add_argument('--cuda', default=None, choices=['cu101', 'latest'], help="CUDA version for torch")
    args = parser.parse_args()

    install_dependencies(cuda_version=args.cuda)
