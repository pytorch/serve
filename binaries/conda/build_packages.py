import os
import sys
import argparse
import subprocess

conda_build_dir = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = subprocess.Popen(
    ['git', 'rev-parse', '--show-toplevel'],
    stdout=subprocess.PIPE
).communicate()[0].rstrip().decode('utf-8')

def conda_build():
    print("## Started torchserve and modelarchiver conda build")

    with open(os.path.join(REPO_ROOT, "ts", "version.txt")) as ts_vf:
        ts_version = ''.join(ts_vf.read().split())
    with open(os.path.join(REPO_ROOT, "model-archiver", "model_archiver", "version.txt")) as ma_vf:
        ma_version = ''.join(ma_vf.read().split())
    with open(os.path.join(REPO_ROOT, "workflow-archiver", "workflow_archiver", "version.txt")) as ma_vf:
        wa_version = ''.join(ma_vf.read().split())

    os.environ["TORCHSERVE_VERSION"] = ts_version
    os.environ["TORCH_MODEL_ARCHIVER_VERSION"] = ma_version
    os.environ["TORCH_WORKFLOW_ARCHIVER_VERSION"] = wa_version
    os.environ["TORCHSERVE_ROOT_DIR"] = REPO_ROOT

    python_versions = ["3.6", "3.7", "3.8"]
    packages = [
        os.path.join(conda_build_dir, pkg)
        for pkg in ["torchserve", "torch-model-archiver", "torch-workflow-archiver"]
    ]

    for pkg in packages:
        for pyv in python_versions:
            output_dir = os.path.join(conda_build_dir, "output")
            cmd = f"conda build --output-folder {output_dir} --python={pyv} {pkg}"
            print(f"## In directory: {os.getcwd()}; Executing command: {cmd}")
            exit_code = os.system(cmd)
            if exit_code != 0:
                print("## Conda Build Failed !")
                return exit_code
    return 0 # Used for sys.exit(0) --> to indicate successful system exit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Conda Build for torchserve and torch-model-archiver")
    args = parser.parse_args()

    conda_build()
