import argparse
import os
import platform
import sys

from print_env_info import run_and_parse_first_match

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(REPO_ROOT)

from ts_scripts.utils import check_python_version


class Common:
    def __init__(self):
        self.torch_stable_url = "https://download.pytorch.org/whl/torch_stable.html"
        self.sudo_cmd = "sudo "

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
                os.system(
                    f"{sys.executable} -m pip install -U -r requirements/torch_{cuda_version}_{platform.system().lower()}.txt"
                )
        else:
            os.system(
                f"{sys.executable} -m pip install -U -r requirements/torch_{platform.system().lower()}.txt"
            )

    def install_python_packages(self, cuda_version, requirements_file_path):
        check = "where" if platform.system() == "Windows" else "which"
        if os.system(f"{check} conda") == 0:
            # conda install command should run before the pip install commands
            # as it may reinstall the packages with different versions
            os.system("conda install -y conda-build")

        self.install_torch_packages(cuda_version)
        os.system(f"{sys.executable} -m pip install -U pip setuptools")
        # developer.txt also installs packages from common.txt
        os.system(f"{sys.executable} -m pip install -U -r {requirements_file_path}")
        # If conda is available install conda-build package

    def install_node_packages(self):
        os.system(
            f"{self.sudo_cmd}npm install -g newman newman-reporter-htmlextra markdown-link-check"
        )

    def install_jmeter(self):
        pass

    def install_wget(self):
        pass


class Linux(Common):
    def __init__(self):
        super().__init__()
        # Skip 'sudo ' when the user is root
        self.sudo_cmd = "" if os.geteuid() == 0 else self.sudo_cmd

        if args.force:
            os.system(f"{self.sudo_cmd}apt-get update")

    def install_java(self):
        if os.system("javac --version") != 0 or args.force:
            os.system(f"{self.sudo_cmd}apt-get install -y openjdk-17-jdk")

    def install_nodejs(self):
        if os.system("node -v") != 0 or args.force:
            os.system(
                f"{self.sudo_cmd}curl -sL https://deb.nodesource.com/setup_14.x | {self.sudo_cmd}bash -"
            )
            os.system(f"{self.sudo_cmd}apt-get install -y nodejs")

    def install_wget(self):
        if os.system("wget --version") != 0 or args.force:
            os.system(f"{self.sudo_cmd}apt-get install -y wget")

    def install_libgit2(self):
        os.system(
            f"wget https://github.com/libgit2/libgit2/archive/refs/tags/v1.3.0.tar.gz -O libgit2-1.3.0.tar.gz"
        )
        os.system(f"tar xzf libgit2-1.3.0.tar.gz")
        os.system(f"cd libgit2-1.3.0 && cmake . && make && sudo make install && cd ..")
        os.system(f"rm -rf libgit2-1.3.0 && rm libgit2-1.3.0.tar.gz")


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


class Darwin(Common):
    def __init__(self):
        super().__init__()

    def install_java(self):
        if os.system("javac -version") != 0 or args.force:
            out = get_brew_version()
            if out == "N/A":
                sys.exit("**Error: Homebrew not installed...")
            os.system("brew install openjdk@17")

    def install_nodejs(self):
        os.system("brew unlink node")
        os.system("brew install node@14")
        os.system("brew link --overwrite node@14")

    def install_node_packages(self):
        os.system(f"{self.sudo_cmd} ./ts_scripts/mac_npm_deps")

    def install_wget(self):
        if os.system("wget --version") != 0 or args.force:
            os.system("brew install wget")


def install_dependencies(cuda_version=None):
    os_map = {"Linux": Linux, "Windows": Windows, "Darwin": Darwin}
    system = os_map[platform.system()]()

    if args.environment == "dev":
        system.install_wget()
        system.install_nodejs()
        system.install_node_packages()

    if platform.system() == "Linux" and args.environment == "dev":
        system.install_libgit2()

    # Sequence of installation to be maintained
    system.install_java()
    requirements_file_path = "requirements/" + (
        "production.txt" if args.environment == "prod" else "developer.txt"
    )
    system.install_python_packages(cuda_version, requirements_file_path)


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
        choices=["cu92", "cu101", "cu102", "cu111", "cu113", "cu116"],
        help="CUDA version for torch",
    )
    parser.add_argument(
        "--environment",
        default="prod",
        choices=["prod", "dev"],
        help="environment(production or developer) on which dependencies will be installed",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="force reinstall dependencies wget, node, java and apt-update",
    )
    args = parser.parse_args()

    install_dependencies(cuda_version=args.cuda)
