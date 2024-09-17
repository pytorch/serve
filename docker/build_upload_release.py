import os
import sys
from argparse import ArgumentParser

# To help discover local modules
REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(REPO_ROOT)

from ts_scripts.utils import check_ts_version, try_and_handle

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
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Delete all built docker images",
    )
    args = parser.parse_args()
    dry_run = args.dry_run
    organization = args.organization

    # Upload pytorch/torchserve docker binaries
    try_and_handle(f"./build_image.sh -m -t {organization}/torchserve:latest", dry_run)
    try_and_handle(
        f"./build_image.sh -g -cv cu121 -t {organization}/torchserve:latest-gpu",
        dry_run,
    )
    try_and_handle(
        f"./build_image.sh -bt dev -cpp -t {organization}/torchserve:latest-cpp-dev-cpu",
        dry_run,
    )
    try_and_handle(
        f"./build_image.sh -bt dev -g -cv cu121 -cpp -t {organization}/torchserve:latest-cpp-dev-gpu",
        dry_run,
    )

    try_and_handle(
        f"docker buildx imagetools create --tag {organization}/torchserve:latest-cpu {organization}/torchserve:latest",
        dry_run,
    )

    try_and_handle(
        f"docker buildx imagetools create --tag {organization}/torchserve:{check_ts_version()}-cpu {organization}/torchserve:latest",
        dry_run,
    )

    try_and_handle(
        f"docker tag {organization}/torchserve:latest-gpu {organization}/torchserve:{check_ts_version()}-gpu",
        dry_run,
    )
    try_and_handle(
        f"docker tag {organization}/torchserve:latest-cpp-dev-cpu {organization}/torchserve:{check_ts_version()}-cpp-dev-cpu",
        dry_run,
    )
    try_and_handle(
        f"docker tag {organization}/torchserve:latest-cpp-dev-gpu {organization}/torchserve:{check_ts_version()}-cpp-dev-gpu",
        dry_run,
    )

    for image in [
        f"{organization}/torchserve:latest-gpu",
        f"{organization}/torchserve:latest-cpp-dev-cpu",
        f"{organization}/torchserve:latest-cpp-dev-gpu",
        f"{organization}/torchserve:{check_ts_version()}-gpu",
        f"{organization}/torchserve:{check_ts_version()}-cpp-dev-cpu",
        f"{organization}/torchserve:{check_ts_version()}-cpp-dev-gpu",
    ]:
        try_and_handle(f"docker push {image}", dry_run)

    # Cleanup built images
    if args.cleanup:
        try_and_handle(f"docker system prune --all --volumes -f", dry_run)
