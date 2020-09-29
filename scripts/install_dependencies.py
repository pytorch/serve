import os
import platform
import argparse


class Common():
    def __init__(self):
        # Assumption is nvidia-smi is installed on systems with gpu
        self.is_gpu_instance = True if os.system("nvidia-smi") == 0 else False
        self.torch_stable_url = "https://download.pytorch.org/whl/torch_stable.html"

    def install_java(self):
        pass

    def install_nodejs(self):
        pass

    def install_torch_packages(self, cu101=False):
        if self.is_gpu_instance:
            if cu101:
                os.system(f"pip install -U -r requirements/torch_cu101.txt -f {self.torch_stable_url}")
            else:
                os.system(f"pip install -U -r requirements/torch.txt -f {self.torch_stable_url}")
        else:
            os.system(f"pip install -U -r requirements/torch_cpu.txt -f {self.torch_stable_url}")

    def install_python_packages(self, cu101=False):
        self.install_torch_packages(cu101=cu101)
        os.system("pip install -U -r requirements/developer.txt") # developer.txt also installs packages from common.txt
        if os.system("conda") == 0: # If conda is available install conda-build package
            os.system("conda install -y conda-build")

    def install_node_packages(self):
        os.system("npm install -g newman newman-reporter-html markdown-link-check")

    def install_jmeter(self):
        pass

    def install_ab(self):
        pass


class Linux(Common):
    def install_java(self):
        os.system("apt-get install -y openjdk-11-jdk")

    def install_nodejs(self):
        os.system("curl -sL https://deb.nodesource.com/setup_14.x | bash -")
        os.system("apt-get update")
        os.system("apt-get install -y nodejs")

    def install_ab(self):
        os.system("apt-get install -y apache2-utils")


class Windows(Common):
    def install_java(self):
        pass

    def install_nodejs(self):
        pass

    def install_ab(self):
        pass


class Darwin(Common):
    def install_java(self):
        os.system("brew tap AdoptOpenJDK/openjdk")
        os.system("brew cask install adoptopenjdk11")

    def install_nodejs(self):
        os.system("brew install node")

    def install_torch_packages(self, cu101=False):
        os.system(f"pip install -U -r requirements/torch.txt -f {self.torch_stable_url}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Install various build and test dependencies of TorchServe")
    parser.add_argument("--java", action="store_true", help="Install Java 11")
    parser.add_argument("--nodejs", action="store_true", help="Install NodeJS")
    parser.add_argument("--python-packages", action="store_true", help="Install Python test packages")
    parser.add_argument("--cu101", action="store_true", help="Install torch packages specific to cu101")
    parser.add_argument("--node-packages", action="store_true", help="Install node packages")
    parser.add_argument("--jmeter", action="store_true", help="Install jmeter")
    parser.add_argument("--ab", action="store_true", help="Install Apache bench")

    args = parser.parse_args()
    os_map = {
        "Linux": Linux,
        "Windows": Windows,
        "Darwin": Darwin
    }
    system = os_map[platform.system()]()

    # Sequence of installation to be maintained
    if args.java:
        system.install_java()
    if args.nodejs:
        system.install_nodejs()
    if args.python_packages:
        system.install_python_packages(cu101=args.cu101)
    if args.node_packages:
        system.install_node_packages()
    if args.jmeter:
        system.install_jmeter()
    if args.ab:
        system.install_ab()
