# Not sure I can assume ts is installed on the dev machine, alternative is reading ts/version.txt
from argparse import ArgumentParser

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
    args = parser.parse_args()
    dry_run = args.dry_run
    organization = args.organization

    try_and_handle(
        f"./build_image.sh -t {organization}/torchserve-kfs:{check_ts_version()}",
        dry_run,
    )
    try_and_handle(
        f"./build_image.sh -g -t {organization}/torchserve-kfs:{check_ts_version()}-gpu",
        dry_run,
    )

    for image in [
        f"{organization}/torchserve-kfs:{check_ts_version()}",
        f"{organization}/torchserve-kfs:{check_ts_version()}-gpu",
    ]:
        try_and_handle(f"docker push {image}", dry_run)
