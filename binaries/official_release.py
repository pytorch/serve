

# THIS IS A DANGEROUS SCRIPT BEWARE! 👻
# TODO: KFP, Conda
# TODO: Simplify the way dry_run works
# TODO: should credentials be environment variables or explicitly passed? (I worry about printing credentials to stdout)
# TODO: Find a way to retag conda binaries
# TODO: Make sure retag.sh script works for pypi

import argparse
from typing import Tuple, List, Dict
import subprocess
import pathlib
CURRENT_FILE_PATH = pathlib.Path(__file__).parent.resolve()
PACKAGES = ["torchserve", "torch-model-archiver", "torch-workflow-archiver"]


def try_and_handle(cmd):
    if DRY_RUN:
        print(f"Executing command: {cmd}")
    else:
        try:
            subprocess.run([cmd], shell=True, check = True)
        except subprocess.CalledProcessError as e:
            raise(e)

def download_nightly_binaries(nightly_date : str) -> Dict[str, List[str]]:
    nightly_binaries = {}

    def download_docker_binaries() -> Tuple[str, str]:
        nightly_cpu = "pytorch/torchserve-nightly:{hardware}-{nightly_date}"
        nightly_gpu = "pytorch/torchserve-nightly:gpu-{nightly_date}"

        for nightly_binary in [nightly_cpu, nightly_gpu]:
            try_and_handle(f"docker pull {nightly_binary}")
        
        return nightly_cpu, nightly_gpu
    
    def download_pypi_binaries() -> List[str]:
        for package in PACKAGES:
            try_and_handle(f"pip download {package}-nightly=={nightly_date}")

        return [f"{package}-nightly-{nightly_date}-py3-none-any.whl" for package in PACKAGES]

    
    # Conda has 3 binaries, one for each OS
    def download_conda_binaries() -> Tuple[str,str,str]:
        # TODO: Conda uses a different date format

        def convert_to_conda_date(date : str) -> str:
            """
            convert from yyyy.mm.dd format to conda format
            """
            return NotImplementedError

        return NotImplementedError
    
    def download_docker_kfp_binaries() -> Tuple[str,str]:
        return NotImplementedError
    
    nightly_binaries = {"docker" : download_docker_binaries(),
                        "pypi" : download_pypi_binaries(),
                        # "conda" : download_conda_binaries(),
                        # "kfp" : download_docker_kfp_binaries(),
                        }
    return nightly_binaries

def tag_binaries(nightly_binaries : Tuple[str, str], official_release_version : str) -> Dict[str, List[str]]:
    binaries_to_promote = {}
    def tag_docker_binaries() -> List[str]:
        nightly_cpu_docker, nightly_gpu_docker = nightly_binaries["docker"]

        cpu_tags = [f"pytorch/torchserve:{tag}" for tag in [f"{official_release_version}-cpu", "latest", "latest-cpu"]]
        gpu_tags = [f"pytorch/torchserve:{tag}" for tag in [f"{official_release_version}-gpu", "latest-gpu"]]

        # CPU tags
        for cpu_tag in cpu_tags:
            try_and_handle(f"docker tag {nightly_cpu_docker} {cpu_tag}")
        
        for gpu_tag in gpu_tags:
            try_and_handle(f"docker tag {nightly_gpu_docker} {gpu_tag}")

        return cpu_tags + gpu_tags

    def tag_pypi_binaries():
        for package, nightly_binary in zip(PACKAGES, nightly_binaries["pypi"]):
            try_and_handle(f"NEW_VERSION={official_release_version} ./{CURRENT_FILE_PATH}/pip/retag_pypi_binary.sh {nightly_binary}")

        # TODO: Check if this naming convention is correct
        # TODO: It is certainly not, need the full name including .whl otherwise can't run twine upload unless I use wildcard matching        
        return [f"{package}_{official_release_version}" for package in PACKAGES]

    def tag_conda_binaries():
        # TODO: Conda also does 

        return NotImplementedError

    def tag_docker_kfp_binaries():
        return NotImplementedError

    binaries_to_promote = {"docker" : tag_docker_binaries(),
                           "pypi" : tag_pypi_binaries(),
                           # "conda" : tag_conda_binaries(),
                           # "kfp" : tag_docker_kfp_binaries(),
                          }

    return binaries_to_promote

# TODO: turn credentials into environment variables instead
def promote_binaries(binaries_to_promote : Dict[str, List[str]], PYPI_CREDENTIALS : str, CONDA_CREDENTIALS : str, DOCKER_CREDENTIALS : str) -> None:
    def promote_docker_binaries():
        try_and_handle(f"docker login --username pytorchbot -password {DOCKER_CREDENTIALS}")

        for binaries in binaries_to_promote["docker"]:
            try_and_handle(f"docker push {binaries}")

    def promote_pypi_binaries():
        for binaries in binaries_to_promote["pypi"]:
            try_and_handle(f"python3 -m twine upload --username __token__ --password {PYPI_CREDENTIALS} {binaries}")

    def promote_conda_binaries():
        return NotImplementedError

    def promote_docker_kfp_binaries():
        return NotImplementedError

    promote_docker_binaries()
    promote_pypi_binaries()
    # promote_conda_binaries()
    # promote_docker_kfp_binaries()

def release(nightly_date : str, official_release_version : str) -> None:
    print("===========")
    print("Starting download nightly binaries")
    nightly_binaries = download_nightly_binaries(nightly_date)
    print("===========")
    print("Completed download nightly binaries")
    print("Started tagging binaries to release version")
    binaries_to_promote = tag_binaries(nightly_binaries, official_release_version)
    print("===========")
    print("Completed tagging binaries to release version")
    print("Started binary promotion")
    promote_binaries(binaries_to_promote)
    print("All binaries have been promoted")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Make an official versioned release - (DO NOT RUN THIS IF YOU DONT KNOW WHAT YOU'RE DOING)"
    )
    parser.add_argument(
        "--nightly_date", type= str, required=True, help="format yyyy.mm.dd"
    )
    parser.add_argument(
        "--official_release_version", type=str, required=True, help="format *.x.y.z"
    )

    parser.add_argument(
        "--pypi_credentials", type=str, required=True, help="Credentials stored in Github secret"
    )

    parser.add_argument(
        "--conda_credentials", type=str, required=False, help="Credentials stored in Github secret"
    )

    parser.add_argument(
        "--docker_credentials", type=str, required=True, help="Credentials stored in Github Secret"
    )

    parser.add_argument(
        "--no_dry_run", action="store_true", required=False, help="dry run is on default since this script is scary"
    )

    args = parser.parse_args()
    print(args)

    global DRY_RUN
    DRY_RUN = not args.no_dry_run

    release(args.nightly_date, args.official_release_version, args.pypi_credentials, args.conda_credentials, args.docker_credentials, args.no_dry_run)