import os
import subprocess
import sys
from argparse import ArgumentParser

# To help discover local modules
REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(REPO_ROOT)


def check_ts_version():
    return "0.8.0"


def try_and_handle(cmd, dry_run=False):
    if dry_run:
        print(f"Executing command: {cmd}")
    else:
        try:
            subprocess.run([cmd], shell=True, check=True)
        except subprocess.CalledProcessError as e:
            raise (e)


def docker_nuke():
    try_and_handle("docker images -q | xargs -r docker rmi -f", dry_run)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--organization",
        type=str,
        default="pytorch",
        help="The name of the Dockerhub organization where the images will be pushed",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="dry_run will print the commands that will be run without running them",
    )
    args = parser.parse_args()
    dry_run = args.dry_run
    organization = args.organization

    # Upload pytorch/torchserve docker binaries
    # GPU
    try_and_handle(
        f"./build_image.sh -g -cv cu117 -t {organization}/torchserve:latest-gpu",
        dry_run,
    )
    try_and_handle(
        f"docker tag {organization}/torchserve:latest-gpu {organization}/torchserve:{check_ts_version()}-gpu",
        dry_run,
    )

    for image in [
        f"{organization}/torchserve:latest-gpu",
        f"{organization}/torchserve:{check_ts_version()}-gpu",
    ]:
        try_and_handle(f"docker push {image}", dry_run)

    # Clean up docker images
    docker_nuke()

    # CPU
    try_and_handle(f"./build_image.sh -t {organization}/torchserve:latest", dry_run)
    try_and_handle(
        f"docker tag {organization}/torchserve:latest {organization}/torchserve:latest-cpu",
        dry_run,
    )
    try_and_handle(
        f"docker tag {organization}/torchserve:latest {organization}/torchserve:{check_ts_version()}-cpu",
        dry_run,
    )
    for image in [
        f"{organization}/torchserve:latest",
        f"{organization}/torchserve:latest-cpu",
        f"{organization}/torchserve:{check_ts_version()}-cpu",
    ]:
        try_and_handle(f"docker push {image}", dry_run)

    # Clean up docker images
    docker_nuke()
