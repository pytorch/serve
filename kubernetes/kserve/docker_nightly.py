import argparse
import os
import sys

# To help discover local modules
REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..")
sys.path.append(REPO_ROOT)

from setup import get_nightly_version
from ts_scripts.utils import try_and_handle

if __name__ == "__main__":
    failed_commands = []
    parser = argparse.ArgumentParser()
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
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Delete all built docker images",
    )
    args = parser.parse_args()
    dry_run = args.dry_run
    organization = args.organization

    project = "torchserve-kfs-nightly"
    cpu_version = f"{project}:cpu-{get_nightly_version()}"
    gpu_version = f"{project}:gpu-{get_nightly_version()}"

    # Build Nightly images and append the date in the name
    try_and_handle(f"./build_image.sh -m -n -t {organization}/{cpu_version}", dry_run)
    try_and_handle(
        f"./build_image.sh -g -n -t {organization}/{gpu_version}",
        dry_run,
    )

    # Push Nightly images to official PyTorch Dockerhub account
    try_and_handle(f"docker push {organization}/{gpu_version}", dry_run)

    # Tag nightly images with latest
    try_and_handle(
        f"docker buildx imagetools create --tag {organization}/{project}:latest-cpu {organization}/{cpu_version}",
        dry_run,
    )
    try_and_handle(
        f"docker tag {organization}/{gpu_version} {organization}/{project}:latest-gpu",
        dry_run,
    )

    # Push images with latest tag
    try_and_handle(f"docker push {organization}/{project}:latest-gpu", dry_run)

    # Cleanup built images
    if args.cleanup:
        try_and_handle(f"docker system prune --all --volumes -f", dry_run)
