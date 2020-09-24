import os
import sys
import argparse


def conda_build_ts(ts_wheel_path, ma_wheel_path):
    python_versions = ["3.6", "3.7", "3.8"]
    packages = ["torchserve", "torch-model-archiver"]

    ts_version = open(os.path.join("..", "..", "ts", "version.txt"), "r").read()
    ma_version = open(os.path.join("..", "..", "model-archiver", "model_archiver", "version.txt"), "r").read()

    os.environ["TORCHSERVE_VERSION"] = ts_version
    os.environ["TORCH_MODEL_ARCHIVER_VERSION"] = ma_version

    os.environ["TORCHSERVE_WHEEL"] = ts_wheel_path
    os.environ["TORCH_MODEL_ARCHIVER_WHEEL"] = ma_wheel_path

    for pkg in packages:
        for pyv in python_versions:
            cmd = f"conda build --output-folder output --python={pyv} {pkg}"
            exit_code = os.system(cmd)
            if exit_code != 0:
                sys.exit("Conda Build Failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Conda Build for torchserve and torch-model-archiver")
    parser.add_argument("--ts-wheel", type=str, required=True, help="torchserve wheel path")
    parser.add_argument("--ma-wheel", type=str, required=True, help="torch-model-archiver wheel path")
    args = parser.parse_args()

    conda_build_ts(args.ts_wheel, args.ma_wheel)
