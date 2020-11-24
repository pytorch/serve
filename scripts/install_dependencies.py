import os
import platform
import argparse


class Common():
    def __init__(self, sudo_cmd):
        # Assumption is nvidia-smi is installed on systems with gpu
        self.is_gpu_instance = True if os.system("nvidia-smi") == 0 else False
        self.torch_stable_url = "https://download.pytorch.org/whl/torch_stable.html"
        self.sudo_cmd = sudo_cmd

    def install_java(self):
        pass

    def install_nodejs(self):
        pass

    def install_torch_packages(self, cuda_version):
        if self.is_gpu_instance:
            if (cuda_version is not None) and cuda_version == 'cu101':
                os.system(f"pip install -U -r requirements/torch_cu101.txt -f {self.torch_stable_url}")
            else:
                os.system(f"pip install -U -r requirements/torch.txt -f {self.torch_stable_url}")
        else:
            os.system(f"pip install -U -r requirements/torch_cpu.txt -f {self.torch_stable_url}")

    def install_python_packages(self, cuda_version):
        self.install_torch_packages(cuda_version)
        os.system("pip install -U -r requirements/developer.txt") # developer.txt also installs packages from common.txt
        if os.system("conda") == 0: # If conda is available install conda-build package
            os.system("conda install -y conda-build")

    def install_node_packages(self):
        os.system(f"{self.sudo_cmd}apt-get update")
        os.system(f"{self.sudo_cmd}npm install -g newman newman-reporter-html markdown-link-check")

    def install_jmeter(self):
        pass


class Linux(Common):
    def __init__(self, sudo_cmd):
        super().__init__(sudo_cmd)

    def install_java(self):
        os.system(f"{self.sudo_cmd}apt-get update")
        os.system(f"{self.sudo_cmd}apt-get install -y openjdk-11-jdk")

    def install_nodejs(self):
        os.system(f"{self.sudo_cmd}apt-get update")
        os.system(f"{self.sudo_cmd}curl -sL https://deb.nodesource.com/setup_14.x | {self.sudo_cmd}bash -")
        os.system(f"{self.sudo_cmd}apt-get install -y nodejs")


class Windows(Common):
    def __init__(self, sudo_cmd):
        super().__init__(sudo_cmd)

    def install_java(self):
        pass

    def install_nodejs(self):
        pass


class Darwin(Common):
    def __init__(self, sudo_cmd):
        super().__init__(sudo_cmd)

    def install_java(self):
        os.system("brew tap AdoptOpenJDK/openjdk")
        os.system("brew cask install adoptopenjdk11")

    def install_nodejs(self):
        os.system("brew install node")

    def install_torch_packages(self, cuda_version=''):
        os.system(f"pip install -U -r requirements/torch.txt -f {self.torch_stable_url}")


def install_dependencies(sudo_cmd='sudo ', cuda_version=None):
    os_map = {
        "Linux": Linux,
        "Windows": Windows,
        "Darwin": Darwin
    }
    system = os_map[platform.system()](sudo_cmd)

    import sys
    # Sequence of installation to be maintained
    system.install_java()
    system.install_nodejs()
    system.install_python_packages(cuda_version)
    system.install_node_packages()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Install various build and test dependencies of TorchServe")
    parser.add_argument('--cuda', default=None, choices=['cu101'], help="CUDA version for torch")
    args = parser.parse_args()

    install_dependencies('', cuda_version=args.cuda)

