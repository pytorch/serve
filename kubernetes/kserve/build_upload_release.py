import os
import sys
from argparse import ArgumentParser

# To help discover local modules
REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..")
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

    try_and_handle(
        f"./build_image.sh -m -t {organization}/torchserve-kfs:{check_ts_version()}",
        dry_run,
    )
    try_and_handle(
        f"./build_image.sh -g -t {organization}/torchserve-kfs:{check_ts_version()}-gpu",
        dry_run,
    )

    # Cleanup built images
    if args.cleanup:
        try_and_handle(f"docker system prune --all --volumes -f", dry_run)
