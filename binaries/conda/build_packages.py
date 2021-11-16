import os
import sys
import argparse
import subprocess

conda_build_dir = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.join(conda_build_dir, "..", "..")
MINICONDA_DOWNLOAD_URL = "https://repo.anaconda.com/miniconda/Miniconda3-py39_4.9.2-Linux-x86_64.sh"
CONDA_BINARY = os.popen("which conda").read().strip() if os.system(f"conda --version") == 0 else  f"$HOME/miniconda/condabin/conda"

def install_conda_build():
    """
    Install conda-build, required to create conda packages
    """
    os.system(f"{CONDA_BINARY} install python=3.8 conda-build anaconda-client -y")

def install_miniconda():
    """
    Installs miniconda, a slimmer anaconda installation to build conda packages
    """

    # Check if conda binary already exists
    exit_code = os.system(f"conda --version")
    if exit_code == 0:
        print(f"'conda' already present on the system. Proceeding without a fresh minconda installation.")
        return

    os.system(f"rm -rf $HOME/miniconda")
    exit_code = os.system(f"wget {MINICONDA_DOWNLOAD_URL} -O ~/miniconda.sh")
    if exit_code != 0:
        print(f"miniconda download failed")
        return exit_code
    os.system(f"bash ~/miniconda.sh -f -b -p $HOME/miniconda")
    os.system(f"echo 'export PATH=$HOME/miniconda/bin:$PATH' >> ~/.bashrc")
    os.system(f"ln -s $HOME/miniconda/bin/activate $HOME/miniconda/condabin/activate")
    os.system(f"ln -s $HOME/miniconda/bin/deactivate $HOME/miniconda/condabin/deactivate")

    os.system(f"{CONDA_BINARY} init")


def conda_build(ts_wheel_path, ma_wheel_path, wa_wheel_path):
    """
    Build conda packages for different python versions
    """

    print("## Started torchserve, model-archiver and workflow-archiver conda build")
    print(f"## Using torchserve wheel: {ts_wheel_path}")
    print(f"## Using model archiver wheel: {ma_wheel_path}")
    print(f"## Using workflow archiver wheel: {wa_wheel_path}")

    with open(os.path.join(REPO_ROOT, "ts", "version.txt")) as ts_vf:
        ts_version = ''.join(ts_vf.read().split())
    with open(os.path.join(REPO_ROOT, "model-archiver", "model_archiver", "version.txt")) as ma_vf:
        ma_version = ''.join(ma_vf.read().split())
    with open(os.path.join(REPO_ROOT, "workflow-archiver", "workflow_archiver", "version.txt")) as wa_vf:
        wa_version = ''.join(wa_vf.read().split())

    os.environ["TORCHSERVE_VERSION"] = ts_version
    os.environ["TORCH_MODEL_ARCHIVER_VERSION"] = ma_version
    os.environ["TORCH_WORKFLOW_ARCHIVER_VERSION"] = wa_version

    os.environ["TORCHSERVE_ROOT_DIR"] = REPO_ROOT.replace("\\", "/")

    os.environ["PYTHON"] = "python"

    python_versions = ["3.6", "3.7", "3.8", "3.9"]
    packages = [
        os.path.join(conda_build_dir, pkg)
        for pkg in ["torchserve", "torch-model-archiver", "torch-workflow-archiver"]
    ]

    for pkg in packages:
        for pyv in python_versions:
            output_dir = os.path.join(conda_build_dir, "output")
            cmd = f"{CONDA_BINARY} build --output-folder {output_dir} --python={pyv} {pkg}"
            print(f"## In directory: {os.getcwd()}; Executing command: {cmd}")
            exit_code = os.system(cmd)
            if exit_code != 0:
                print("## Conda Build Failed !")
                return exit_code
    return 0 # Used for sys.exit(0) --> to indicate successful system exit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Conda Build for torchserve, torch-model-archiver and torch-workflow-archiver")
    parser.add_argument("--ts-wheel", type=str, required=False, help="torchserve wheel path")
    parser.add_argument("--ma-wheel", type=str, required=False, help="torch-model-archiver wheel path")
    parser.add_argument("--wa-wheel", type=str, required=False, help="torch-workflow-archiver wheel path")
    parser.add_argument("--install-conda-dependencies", action="store_true", required=False, help="specify to install miniconda and conda-build")
    args = parser.parse_args()
    
    if args.install_conda_dependencies:
        install_miniconda()
        install_conda_build()
        
    if all([args.ts_wheel, args.ma_wheel, args.wa_wheel]):
        conda_build(args.ts_wheel, args.ma_wheel, args.wa_wheel)
