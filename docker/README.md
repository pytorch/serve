## Contents of this Document

* [Prerequisites](#docker_prerequisite)
* [Create TorchServe docker image](#docker_image_production)
* [Create TorchServe docker image from source](#docker_image_source)

# Prerequisites

* docker - Refer to the [official docker installation guide](https://docs.docker.com/install/)
* git    - Refer to the [official git set-up guide](https://help.github.com/en/github/getting-started-with-github/set-up-git)
* TorchServe source code. Clone and enter the repo as follows:

```bash
git clone https://github.com/pytorch/serve.git
cd serve
```

# Create TorchServe docker image

For creating CPU based image :
```bash
docker build --file Dockerfile.cpu -t torchserve:latest .
```

For creating GPU based image :
```bash
docker build --file Dockerfile.gpu -t torchserve:latest .
```

## Start a container with a TorchServe image

The following examples will start the container with 8080/81 port exposed to outer-world/localhost.

### Start CPU container

For specific versions you can pass in the specific tag to use (ex: 0.1-cpu):
```bash
docker run --rm -it -p 8080:8080 -p 8081:8081 pytorch/torchserve:0.1-cpu
```
For GPU based image :

```bash
docker run --rm -it --gpus --gpus '"device=1,2"' -p 8080:8080 -p 8081:8081 torchserve:1.0
```

For the latest version, you can use the `latest` tag:
docker run --rm -it -p 8080:8080 -p 8081:8081 pytorch/torchserve:latest

#### Start GPU container

For specific versions you can pass in the specific tag to use (ex: 0.1-cpu):
```bash
docker run --rm -it --gpus all -p 8080:8080 -p 8081:8081 pytorch/torchserve:0.1-cuda10.1-cudnn7-runtime
```

For the latest version, you can use the `gpu-latest` tag:
```bash
docker run --rm -it --gpus all -p 8080:8080 -p 8081:8081 pytorch/torchserve:latest-gpu
```

#### Accessing TorchServe APIs inside container

The TorchServe's inference and management APIs can be accessed on localhost over 8080 and 8081 ports respectively. Example :

```bash
curl http://localhost:8080/ping
```

# Create TorchServe docker image from source

The following are examples on how to use the `build_image.sh` script to build Docker images to support CPU or GPU inference.

To build the TorchServe image for a CPU device using the `master` branch, use the following command:

```bash
./build_image.sh
```

To create a Docker image for a specific branch, use the following command:

```bash
./build_image.sh -b <branch_name>
```

To create a Docker image for a specific branch and specific tag, use the following command:

```bash
./build_image.sh -b <branch_name> -t <tagname:latest>
```

To create a Docker image for a GPU device, use the following command:

```bash
./build_image.sh --gpu
```

To create a Docker image for a GPU device with a specific branch, use following command:

```bash
./build_image.sh -b <branch_name> --gpu
```

To run your TorchServe Docker image and start TorchServe inside the container with a pre-registered `resnet-18` image classification model, use the following command:

```bash
./start.sh
```
For GPU run the following command:
```bash
./start.sh --gpu
```
For GPU with specific GPU device ids run the following command:
```bash
./start.sh --gpu_devices 1,2,3
```
