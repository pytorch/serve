# Not sure I can assume ts is installed on the dev machine, alternative is reading ts/version.txt
import subprocess
from argparse import ArgumentParser

from ts import version


# Move this to common utils function?
def try_and_handle(cmd, dry_run=False):
    if dry_run:
        print(f"Executing command: {cmd}")
    else:
        try:
            subprocess.run([cmd], shell=True, check=True)
        except subprocess.CalledProcessError as e:
            raise (e)


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
        f"./build_image.sh -t {organization}/torchserve-kfs:{version()}", dry_run
    )
    try_and_handle(
        f"./build_image.sh -g -t {organization}/torchserve-kfs:{version()}-gpu", dry_run
    )

    for image in [
        f"{organization}/torchserve-kfs:{version()}",
        f"{organization}/torchserve-kfs:{version()}-gpu",
    ]:
        try_and_handle(f"docker push {image}", dry_run)
