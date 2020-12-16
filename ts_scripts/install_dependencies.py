import os
import platform
import argparse
import sys
from pathlib import Path


REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(REPO_ROOT)

from ts_scripts.utils import check_python_version


class Common():
    def __init__(self):
        self.torch_stable_url = "https://download.pytorch.org/whl/torch_stable.html"
        self.sudo_cmd = 'sudo '

    def install_java(self):
        pass

    def install_nodejs(self):
        pass

    def install_torch_packages(self, cuda_version):
        if cuda_version is None:
            os.system(f"pip install -U -r requirements/torch_cpu.txt -f {self.torch_stable_url}")
        elif cuda_version and cuda_version != "latest":
            os.system(f"pip install -U -r requirements/torch_{cuda_version}.txt -f {self.torch_stable_url}")
        else:
            os.system(f"pip install -U -r requirements/torch.txt")

    def install_python_packages(self, cuda_version, requirements_file_path):
        if os.system("conda") == 0:
            # conda install command should run before the pip install commands
            # as it may reinstall the packages with different versions
            os.system("conda install -y conda-build")

        self.install_torch_packages(cuda_version)
        os.system("pip install -U pip setuptools")
        # developer.txt also installs packages from common.txt
        os.system("pip install -U -r {0}".format(requirements_file_path))
        # If conda is available install conda-build package


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
        python_path = Path(sys.executable).resolve()
        os.system(f"{self.sudo_cmd}curl -sL https://deb.nodesource.com/setup_14.x | {self.sudo_cmd}bash -")
        os.system(f"{self.sudo_cmd}apt-get install -y nodejs")
        os.system(f"{self.sudo_cmd}ln -sf {python_path} /usr/bin/python")
        os.system(f"{self.sudo_cmd}ln -sf /usr/bin/pip3 /usr/bin/pip")


class Windows(Common):
    def __init__(self):
        super().__init__()
        self.sudo_cmd = ''

    def install_torch_packages(self, cuda_version):
        if cuda_version is None:
            os.system(f"pip install -U -r requirements/torch_cpu.txt -f {self.torch_stable_url}")
        elif cuda_version and cuda_version != "latest":
            os.system(f"pip install -U -r requirements/torch_{cuda_version}.txt -f {self.torch_stable_url}")
        else:
            os.system(f"pip install -U -r requirements/torch.txt -f {self.torch_stable_url}")

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

    def install_node_packages(self):
        os.system(f"{self.sudo_cmd} ./ts_scripts/mac_npm_deps")


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
    requirements_file_path = "requirements/" + ("production.txt" if args.environment == "prod" else "developer.txt")
    system.install_python_packages(cuda_version, requirements_file_path)

    if args.environment == "dev":
        system.install_nodejs()
        system.install_node_packages()


if __name__ == "__main__":
    check_python_version()
    parser = argparse.ArgumentParser(description="Install various build and test dependencies of TorchServe")
    parser.add_argument('--cuda', default=None, choices=['cu92', 'cu101', 'cu102',  'latest'], help="CUDA version for torch")
    parser.add_argument('--environment', default='prod', choices=['prod', 'dev'],
                        help="environment(production or developer) on which dependencies will be installed")

    args = parser.parse_args()

    install_dependencies(cuda_version=args.cuda)
