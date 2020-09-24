import os
import sys
import argparse

conda_build_dir = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.join(conda_build_dir, "..", "..")


def conda_build(ts_wheel_path, ma_wheel_path):
    print("## Started torchserve and modelarchiver conda build")
    print(f"## Using torchserve wheel: {ts_wheel_path}")
    print(f"## Using model archiver wheel: {ma_wheel_path}")

    with open(os.path.join(REPO_ROOT, "ts", "version.txt")) as ts_vf:
        ts_version = ''.join(ts_vf.read().split())
    with open(os.path.join(REPO_ROOT, "model-archiver", "model_archiver", "version.txt")) as ma_vf:
        ma_version = ''.join(ma_vf.read().split())

    os.environ["TORCHSERVE_VERSION"] = ts_version
    os.environ["TORCH_MODEL_ARCHIVER_VERSION"] = ma_version

    os.environ["TORCHSERVE_WHEEL"] = ts_wheel_path
    os.environ["TORCH_MODEL_ARCHIVER_WHEEL"] = ma_wheel_path

    python_versions = ["3.6", "3.7", "3.8"]
    packages = [os.path.join(conda_build_dir, "torchserve"), os.path.join(conda_build_dir, "torch-model-archiver")]

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
    parser.add_argument("--ts-wheel", type=str, required=True, help="torchserve wheel path")
    parser.add_argument("--ma-wheel", type=str, required=True, help="torch-model-archiver wheel path")
    args = parser.parse_args()

    conda_build(args.ts_wheel, args.ma_wheel)
