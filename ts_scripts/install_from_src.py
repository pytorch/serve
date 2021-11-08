import os
import sys
import time
import shutil


# To help discover local modules
REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(REPO_ROOT)

from ts_scripts import print_env_info as build_hdr_printer
from ts_scripts.utils import check_python_version
from ts_scripts.utils import is_conda_env


def clean_slate(): 
    print("## Uninstall existing torchserve and model archiver")
    if is_conda_env():
        cmd = "conda uninstall -y torchserve torch-model-archiver workflow-model-archiver"
    else:
        cmd = "pip uninstall -y torchserve torch-model-archiver workflow-model-archiver"
    print(f"## In directory: {os.getcwd()} | Executing command: {cmd}")
    os.system(cmd)
    time.sleep(5)


def install_torchserve():
    print("## Install torchserve from source")
    cmd = "pip install ."
    print(f"## In directory: {os.getcwd()} | Executing command: {cmd}")
    os.system(cmd)


def install_torch_model_archiver():
    print("## Install torch-model-archiver from source")
    cmd = "pip install model-archiver/."
    print(f"## In directory: {os.getcwd()} | Executing command: {cmd}")
    os.system(cmd)


def install_torch_workflow_archiver():
    print("## Install torch-workflow-archiver from source")
    cmd = "pip install workflow-archiver/."
    print(f"## In directory: {os.getcwd()} | Executing command: {cmd}")
    os.system(cmd)


def clean_up_build_residuals():
    print("## Cleaning build residuals (__pycache__)")
    try:
        for (dirpath, dirnames, filenames) in os.walk(os.getcwd()):
            if "__pycache__" in dirnames:
                cache_dir = os.path.join(dirpath, "__pycache__")
                print(f"## Removing - {cache_dir}")
                shutil.rmtree(cache_dir)
    except Exception as e:
        print(f"#Error while cleaning cache file. Details - {str(e)}")


def install_from_src():
    clean_slate()
    install_torch_model_archiver()
    install_torch_workflow_archiver()
    install_torchserve()    
    clean_up_build_residuals()


if __name__ == '__main__':
    check_python_version()
    from pygit2 import Repository
    git_branch = Repository('.').head.shorthand
    build_hdr_printer.main(git_branch)
    install_from_src()