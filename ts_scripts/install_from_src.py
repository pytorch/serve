import os
import sys
import time
import shutil


# To help discover local modules
REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(REPO_ROOT)

from ts_scripts import tsutils as ts
from ts_scripts import install_dependencies as idep


def clean_slate(): 
    print("## Uninstall existing torchserve and model archiver")
    if ts.is_conda_env():
        cmd = "conda uninstall -y torchserve torch-model-archiver"
    else:
        cmd = "pip uninstall -y torchserve torch-model-archiver"
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
    install_torchserve()    
    clean_up_build_residuals()


if __name__ == '__main__':
    idep.install_dependencies()
    install_from_src()
