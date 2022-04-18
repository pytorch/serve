from datetime import date
import os

def get_nightly_version():
    today = date.today()
    return today.strftime("%Y.%m.%d")

if __name__ == "__main__":
    project = "torchserve-nightly"
    cpu_version = f"{project}:cpu-{get_nightly_version()}"
    gpu_version = f"{project}:gpu-{get_nightly_version()}"

    # Build Nightly images and append the date in the name
    os.system(f"./build_image.sh -bt dev -t pytorch/{cpu_version}")
    os.system(f"./build_image.sh -bt dev -g -cv cu102 -t pytorch/{gpu_version}")

    # Push Nightly images to official PyTorch Dockerhub account
    for version in [cpu_version, gpu_version]:
        os.system(f"docker push pytorch/{version}")
    
    # Tag images with latest and push those as well
    os.system(f"docker tag pytorch/{cpu_version} pytorch/{project}:latest-cpu")
    os.system(f"docker tag pytorch/{gpu_version} pytorch/{project}:latest-gpu")

    for version in ["latest-cpu", "latest-gpu"]:
        os.system(f"docker push pytorch/{project}:{version}")
