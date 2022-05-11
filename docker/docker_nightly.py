from datetime import date
import os
import subprocess


def get_nightly_version():
    today = date.today()
    return today.strftime("%Y.%m.%d")

if __name__ == "__main__":
    project = "torchserve-nightly"
    cpu_version = f"{project}:cpu-{get_nightly_version()}"
    gpu_version = f"{project}:gpu-{get_nightly_version()}"

    # Build Nightly images and append the date in the name
    try:
        subprocess.run([f"./build_image.sh -bt dev -t pytorch/{cpu_version}"], check = True)
    except subprocess.CalledProcessError:
        print("Docker CPU build has failed")
    
    try:
        subprocess.run([f"/build_image.sh -bt dev -t pytorch/{cpu_version}"], check = True)
    except subprocess.CalledProcessError:
        print("Docker GPU build has failed")


    # Push Nightly images to official PyTorch Dockerhub account
    for version in [cpu_version, gpu_version]:
        os.system(f"docker push pytorch/{version}")
    
    # Tag images with latest and push those as well
    os.system(f"docker tag pytorch/{cpu_version} pytorch/{project}:latest-cpu")
    os.system(f"docker tag pytorch/{gpu_version} pytorch/{project}:latest-gpu")

    for version in ["latest-cpu", "latest-gpu"]:
        try:
            os.system(f"docker push pytorch/{project}:{version}")
        except subprocess.CalledProcessError:
            print(f"Docker push {version} failed")

