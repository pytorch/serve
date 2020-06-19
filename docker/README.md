## Contents of this Document

* [Prerequisites](#prerequisites)
* [Create TorchServe docker image](#create-torchserve-docker-image)
* [Create TorchServe docker image from source](#create-torchserve-docker-image-from-source)
* [Create torch-model-archiver from container](#create-torch-model-archiver-from-container)
* [Running TorchServe docker image in production](#running-torchserve-in-a-production-docker-environment)

# Prerequisites

* docker - Refer to the [official docker installation guide](https://docs.docker.com/install/)
* git    - Refer to the [official git set-up guide](https://help.github.com/en/github/getting-started-with-github/set-up-git)
* For base Ubuntu with GPU, install following nvidia container toolkit and driver- 
  * [Nvidia container toolkit](https://github.com/NVIDIA/nvidia-docker#ubuntu-160418042004-debian-jessiestretchbuster)
  * [Nvidia driver](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/install-nvidia-driver.html)

## First things first

```bash
1. If you have not clone torchserve source then:
git clone https://github.com/pytorch/serve.git
2. cd serve/docker
```

# Create TorchServe docker image

For creating CPU based image :
```bash
DOCKER_BUILDKIT=1 docker build --file Dockerfile -t torchserve:latest .
```

For creating GPU based image :
```bash
DOCKER_BUILDKIT=1 docker build --file Dockerfile --build-arg BASE_IMAGE=nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04 -t torchserve:latest .
```

## Start a container with a TorchServe image
The following examples will start the container with 8080/81 port exposed to outer-world/localhost.

#### Start CPU container

For the latest version, you can use the `latest-cpu` tag:
```bash
docker run --rm -it -p 8080:8080 -p 8081:8081 pytorch/torchserve:latest-cpu
```

For specific versions you can pass in the specific tag to use (ex: pytorch/torchserve:0.1.1-cpu):
```bash
docker run --rm -it -p 8080:8080 -p 8081:8081 pytorch/torchserve:0.1.1-cpu
```

#### Start GPU container

For GPU latest image with gpu devices 1 and 2:
```bash
docker run --rm -it --gpus '"device=1,2"' -p 8080:8080 -p 8081:8081 pytorch/torchserve:latest-gpu
```

For specific versions you can pass in the specific tag to use (ex: 0.1.1-cuda10.1-cudnn7-runtime):
```bash
docker run --rm -it --gpus all -p 8080:8080 -p 8081:8081 pytorch/torchserve:0.1.1-cuda10.1-cudnn7-runtime
```

For the latest version, you can use the `latest-gpu` tag:
```bash
docker run --rm -it --gpus all -p 8080:8080 -p 8081:8081 pytorch/torchserve:latest-gpu
```

#### Accessing TorchServe APIs inside container

The TorchServe's inference and management APIs can be accessed on localhost over 8080 and 8081 ports respectively. Example :

```bash
curl http://localhost:8080/ping
```

# Create TorchServe docker image from source

The following are examples on how to use the `build_image.sh` script to build Docker images from source to support CPU or GPU inference.

To build the TorchServe image for a CPU device using the `master` branch, use the following command:

```bash
./build_image.sh
```

Alternatively, you can use following direct command, (assuming you have followed steps in [Clone serve source](#first-things-first))- 
```bash 
1. Do one more clone -> `git clone https://github.com/pytorch/serve.git`
2. cd serve;git checkout <branch>;cd ..

For cpu -
3. DOCKER_BUILDKIT=1 docker build --file Dockerfile_dev.cpu -t torchserve:dev .

For gpu - 
3. DOCKER_BUILDKIT=1 docker build --file Dockerfile_dev.gpu -t torchserve:dev .
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
Alternatively, you can use direct commands describe in [Start a container with a TorchServe image](#start-a-container-with-a-torchserve-image) above for cpu and gpu by changing image name

# Create torch-model-archiver from container

To create mar [model archive] file for torchserve deployment, you can use following steps

1. Start container by sharing your local model-store/any directory containing custom/example mar contents as well as model-store directory (if not there, create it)

```bash
docker run --rm -it -p 8080:8080 -p 8081:8081 --name mar -v $(pwd)/model-store:/home/model-server/model-store -v $(pwd)/examples:/home/model-server/examples  torchserve:latest
```

1. List your container or skip this if you know cotainer name 
```bash
docker ps
```

2. Bind and get the bash prompt of running container
```bash
docker exec -it <container_name> /bin/bash
```
You will be landing at /home/model-server/.

3. Now Execute torch-model-archiver command e.g.
```bash
torch-model-archiver --model-name densenet161 --version 1.0 --model-file /home/model-server/examples/image_classifier/densenet_161/model.py --serialized-file /home/model-server/examples/image_classifier/densenet161-8d451a50.pth --export-path /home/model-server/model-store --extra-files /home/model-server/examples/image_classifier/index_to_name.json --handler image_classifier
```
Refer [torch-model-archiver](../model-archiver/README.md) for details.

4. desnet161.mar file should be present at /home/model-server/model-store

# Running TorchServe in a Production Docker Environment.

You may want to consider the following aspects / docker options when deploying torchserve in Production with Docker.


* Shared Memory Size 

    * ```shm-size``` - The shm-size parameter allows you to specify the shared memory that a container can use. It enables memory-intensive containers to run faster by giving more access to allocated memory.


* User Limits for System Resources
    
    * ```--ulimit memlock=-1``` : Maximum locked-in-memory address space. 
    * ```--ulimit stack``` : Linux stack size 

    The current ulimit values can be viewed by executing ```ulimit -a```. A more exhaustive set of options for resource constraining can be found in the Docker Documentation [here](https://docs.docker.com/config/containers/resource_constraints/), [here](https://docs.docker.com/engine/reference/commandline/run/#set-ulimits-in-container---ulimit) and [here](https://docs.docker.com/engine/reference/run/#runtime-constraints-on-resources)


* Exposing specific ports / volumes between the host & docker env.

    *  ```-p8080:p8080 -p8081:8081``` TorchServe uses default ports 8080 / 8081 for inference & management APIs. You may want to expose these ports to the host for HTTP Requests between Docker & Host.
    * The model store is passed to torchserve with the --model-store option. You may want to consider using a shared volume if you prefer pre populating models in model-store directory.

For example,

```
docker run --rm --shm-size=1g \
        --ulimit memlock=-1 \
        --ulimit stack=67108864 \
        -p8080:8080 \
        -p8081:8081 \
        --mount type=bind,source=/path/to/model/store,target=/tmp/models <container> torchserve --model-store=/tmp/models 
```
