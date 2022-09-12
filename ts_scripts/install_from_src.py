import argparse
import os
import sys

# To help discover local modules
REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(REPO_ROOT)

from ts_scripts import print_env_info as build_hdr_printer
from ts_scripts.utils import check_python_version


def install_from_src(dev=False):
    for binary in [".", "model-archiver/.", "workflow-archiver/."]:
        cmd = (
            f"pip install --force-reinstall -e {binary}"
            if dev
            else f"pip install --force-reinstall {binary}"
        )
        print(f"## In directory {os.getcwd()} | Executing command {cmd}")
        os.system(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--environment",
        type=str,
        default="production",
        help="options: dev|prod",
    )
    args = parser.parse_args()
    check_python_version()
    from pygit2 import Repository

    git_branch = Repository(".").head.shorthand
    build_hdr_printer.main(git_branch)
    install_from_src(args.environment == "dev")
