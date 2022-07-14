from .docker_nightly import try_and_handle
from ts import version


if __name__ == "__main__":

    # Upload pytorch/torchserve docker binaries
    try_and_handle("./build_image.sh -bt dev -g -cv -t pytorch/torchserve:latest")
    try_and_handle("./build_image.sh -bt dev -g -cv 102 -t pytorch/torchserve:latest-gpu")
    try_and_handle("docker tag pytorch/torchserve-latest pytorch/torchserve:latest-cpu")
    try_and_handle(f"docker tag pytorch/torchserve:latest pytorch/torchserve:{version()}-cpu")
    try_and_handle(f"docker tag pytorch/torchserve:latest pytorch/torchserve:{version()}-gpu")

    for image in ["pytorch/torchserve:latest", "pytorch/torchserve:latest-cpu", "pytorch/torchserve:latest-gpu", f"pytorch/torchserve:{version()}-cpu", f"pytorch/torchserve:{version()}-gpu"]:
        try_and_handle(f"docker push {image}")