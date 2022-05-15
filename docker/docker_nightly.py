from datetime import date
import os
import sys

def get_nightly_version():
    today = date.today()
    return today.strftime("%Y.%m.%d")

if __name__ == "__main__":
    failed_commands = []

    def try_and_handle(cmd, dry_run = False):
        if dry_run:
            print(f"Executing command: {cmd}")
        else:
            code = os.system(cmd)
            if code != 0:
                failed_commands.append(cmd)
    
    dry_run = sys.argv
    
    project = "torchserve-nightly"
    cpu_version = f"{project}:cpu-{get_nightly_version()}"
    gpu_version = f"{project}:gpu-{get_nightly_version()}"

    # Build Nightly images and append the date in the name
    try_and_handle(f"./build_image.sh -bt dev -t pytorch/{cpu_version}", dry_run)
    try_and_handle(f"./build_image.sh -bt dev -g -cv cu102 -t pytorch/{gpu_version}", dry_run)

    # Push Nightly images to official PyTorch Dockerhub account
    try_and_handle(f"docker push pytorch/{cpu_version}", dry_run)
    try_and_handle(f"docker push pytorch/{gpu_version}", dry_run)

    
    # Tag nightly images with latest
    try_and_handle(f"docker tag pytorch/{cpu_version} pytorch/{project}:latest-cpu", dry_run)
    try_and_handle(f"docker tag pytorch/{gpu_version} pytorch/{project}:latest-gpu", dry_run)

    # Push images with latest
    try_and_handle(f"docker push pytorch/{project}:{latest-cpu}", dry_run)
    try_and_handle(f"docker push pytorch/{project}:{latest-gpu}", dry_run)
    
    # If there are any errors raise them but don't block commands that succeeded
    if failed_commands:
        raise Exception(f"failed commands: {failed_commands})
