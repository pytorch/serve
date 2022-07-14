from argparse import ArgumentParser

from ts import version
from ts_scripts.utils import try_and_handle

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
    try_and_handle(
        f"./build_image.sh -bt dev -t {organization}/torchserve:latest", dry_run
    )
    try_and_handle(
        f"./build_image.sh -bt dev -g -cv 102 -t {organization}/torchserve:latest-gpu",
        dry_run,
    )
    try_and_handle(
        f"docker tag {organization}/torchserve-latest {organization}/torchserve:latest-cpu",
        dry_run,
    )
    try_and_handle(
        f"docker tag {organization}/torchserve:latest {organization}/torchserve:{version()}-cpu",
        dry_run,
    )
    try_and_handle(
        f"docker tag {organization}/torchserve:latest-gpu {organization}/torchserve:{version()}-gpu",
        dry_run,
    )

    for image in [
        f"{organization}/torchserve:latest",
        f"{organization}/torchserve:latest-cpu",
        f"{organization}/torchserve:latest-gpu",
        f"{organization}/torchserve:{version()}-cpu",
        f"{organization}/torchserve:{version()}-gpu",
    ]:
        try_and_handle(f"docker push {image}")
