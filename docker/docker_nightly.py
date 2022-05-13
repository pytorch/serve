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
    os.system(f"docker push pytorch/{cpu_version}")
    os.system(f"docker push pytorch/{gpu_version}")

    
    # Tag nightly images with latest
    os.system(f"docker tag pytorch/{cpu_version} pytorch/{project}:latest-cpu")
    os.system(f"docker tag pytorch/{gpu_version} pytorch/{project}:latest-gpu")

    # Push images with latest
    os.system(f"docker push pytorch/{project}:{latest-cpu}")
    os.system(f"docker push pytorch/{project}:{latest-gpu}")
