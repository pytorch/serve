import os
import platform
import time
import argparse
import shutil
import install_dependencies


def clean_slate(): 
    print("Cleaning up state")
    os.system('pip uninstall --yes torchserve || true')
    os.system('pip uninstall --yes torch-model-archiver || true')
    time.sleep(5)


def install_torchserve():
    print("Install torchserve from source")
    os.system('pip install .')


def install_torch_model_archiver():
    print("Install torch-model-archiver from source")
    os.system(f"pip install model-archiver/.")


def clean_up_build_residuals(): 
    try:
        for (dirpath, dirnames, filenames) in os.walk(os.getcwd()):
            if '__pycache__' in dirnames:
                cache_dir = os.path.join(dirpath, '__pycache__')
                print(f'Cleaning - {cache_dir}')
                shutil.rmtree(cache_dir)
    except Exception as e:
        print('Error while cleaning cache file. Details - '+str(e))


def install_from_src(cu101=False):
    clean_slate()
    system = getattr(install_dependencies, platform.system())()
    system.install_python_packages(cu101=cu101)
    install_torchserve()
    install_torch_model_archiver()
    clean_up_build_residuals()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Install TorchServe from source")
    parser.add_argument("--cu101", action="store_true", help="Install torch packages specific to cu101")
    args = parser.parse_args()
    install_from_src(cu101=args.cu101)
