## Contents of this Document

* [Prerequisites](#prerequisites)
* [Create TorchServe docker image](#create-torchserve-docker-image)
* [Create torch-model-archiver from container](#create-torch-model-archiver-from-container)
* [Running TorchServe docker image in production](#running-torchserve-in-a-production-docker-environment)

# Prerequisites

* docker - Refer to the [official docker installation guide](https://docs.docker.com/install/)
* git    - Refer to the [official git set-up guide](https://help.github.com/en/github/getting-started-with-github/set-up-git)
* For base Ubuntu with GPU, install following nvidia container toolkit and driver-
  * [Nvidia container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#installing-on-ubuntu-and-debian)
  * [Nvidia driver](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/install-nvidia-driver.html)
  
* NOTE - Dockerfiles have not been tested on windows native platform.

## First things first

If you have not cloned TorchServe source then:
```bash
git clone https://github.com/pytorch/serve.git
cd serve/docker
```

# Create TorchServe docker image

Use `build_image.sh` script to build the docker images. The script builds the `production`, `dev` and `codebuild` docker images.
| Parameter | Desciption |
|------|------|
|-h, --help|Show script help|
|-b, --branch_name|Specify a branch name to use. Default: master |
|-g, --gpu|Build image with GPU based ubuntu base image|
|-bt, --buildtype|Which type of docker image to build. Can be one of : production, dev, codebuild|
|-t, --tag|Tag name for image. If not specified, script uses torchserve default tag names.|
|-cv, --cudaversion| Specify to cuda version to use. Supported values `cu92`, `cu101`, `cu102`, `cu111`, `cu113`. Default `cu102`|
|--codebuild| Set if you need [AWS CodeBuild](https://aws.amazon.com/codebuild/)|


**PRODUCTION ENVIRONMENT IMAGES**

Creates a docker image with publicly available `torchserve` and `torch-model-archiver` binaries installed.

 - To create a CPU based image

```bash
./build_image.sh
```

 - To create a GPU based image with cuda 10.2. Options are `cu92`, `cu101`, `cu102`, `cu111`, `cu113`

  ```bash
  ./build_image.sh -g -cv cu102
  ```

 - To create an image with a custom tag

```bash
./build_image.sh -t torchserve:1.0
```

**DEVELOPER ENVIRONMENT IMAGES**

Creates a docker image with `torchserve` and `torch-model-archiver` installed from source.

- For creating CPU based image :

```bash
./build_image.sh -bt dev
```

- For creating CPU based image with a different branch:

```bash
./build_image.sh -bt dev -b my_branch
```


- For creating GPU based image with cuda version 11.3:

```bash
./build_image.sh -bt dev -g -cv cu113
```

- For creating GPU based image with cuda version 11.1:

```bash
./build_image.sh -bt dev -g -cv cu111
```

- For creating GPU based image with cuda version 10.2:

```bash
./build_image.sh -bt dev -g -cv cu102
```

 - For creating GPU based image with cuda version 10.1:

```bash
./build_image.sh -bt dev -g -cv cu101
```

 - For creating GPU based image with cuda version 9.2:

```bash
./build_image.sh -bt dev -g -cv cu92
```

- For creating GPU based image with a different branch:

```bash
./build_image.sh -bt dev -g -cv cu113 -b my_branch
```

```bash
./build_image.sh -bt dev -g -cv cu111 -b my_branch
```

 - For creating image with a custom tag:

```bash
./build_image.sh -bt dev -t torchserve-dev:1.0
```

**CODEBUILD ENVIRONMENT IMAGES**

Creates a docker image for codebuild environment

- For creating CPU based image :

```bash
./build_image.sh -bt codebuild
```

- For creating GPU based image with cuda version 11.1:

```bash
./build_image.sh -bt codebuild -g -cv cu113
```

```bash
./build_image.sh -bt codebuild -g -cv cu111
```

 - For creating GPU based image with cuda version 10.1:

```bash
./build_image.sh -bt codebuild -g -cv cu101
```

 - For creating GPU based image with cuda version 9.2:

```bash
./build_image.sh -bt codebuild -g -cv cu92
```

 - For creating image with a custom tag:

```bash
./build_image.sh -bt codebuild -t torchserve-codebuild:1.0
```


## Start a container with a TorchServe image

The following examples will start the container with 8080/81/82 and 7070/71 port exposed to outer-world/localhost.

#### Start CPU container

For the latest version, you can use the `latest` tag:

```bash
docker run --rm -it -p 8080:8080 -p 8081:8081 -p 8082:8082 -p 7070:7070 -p 7071:7071 pytorch/torchserve:latest
```

For specific versions you can pass in the specific tag to use (ex: pytorch/torchserve:0.1.1-cpu):

```bash
docker run --rm -it -p 8080:8080 -p 8081:8081 -p 8082:8082 -p 7070:7070 -p 7071:7071  pytorch/torchserve:0.1.1-cpu
```

#### Start GPU container

For GPU latest image with gpu devices 1 and 2:

```bash
docker run --rm -it --gpus '"device=1,2"' -p 8080:8080 -p 8081:8081 -p 8082:8082 -p 7070:7070 -p 7071:7071 pytorch/torchserve:latest-gpu
```

For specific versions you can pass in the specific tag to use (ex: `0.1.1-cuda10.1-cudnn7-runtime`):

```bash
docker run --rm -it --gpus all -p 8080:8080 -p 8081:8081 -p 8082:8082 -p 7070:7070 -p 7071:7071 pytorch/torchserve:0.1.1-cuda10.1-cudnn7-runtime
```

For the latest version, you can use the `latest-gpu` tag:

```bash
docker run --rm -it --gpus all -p 8080:8080 -p 8081:8081 -p 8082:8082 -p 7070:7070 -p 7071:7071 pytorch/torchserve:latest-gpu
```

#### Accessing TorchServe APIs inside container

The TorchServe's inference and management APIs can be accessed on localhost over 8080 and 8081 ports respectively. Example :

```bash
curl http://localhost:8080/ping
```

# Create torch-model-archiver from container

To create mar [model archive] file for TorchServe deployment, you can use following steps

1. Start container by sharing your local model-store/any directory containing custom/example mar contents as well as model-store directory (if not there, create it)

```bash
docker run --rm -it -p 8080:8080 -p 8081:8081 --name mar -v $(pwd)/model-store:/home/model-server/model-store -v $(pwd)/examples:/home/model-server/examples  pytorch/torchserve:latest
```

1. List your container or skip this if you know container name

```bash
docker ps
```

2. Bind and get the bash prompt of running container

```bash
docker exec -it <container_name> /bin/bash
```

You will be landing at /home/model-server/.

3. Download the model weights if you have not done so already (they are not part of the repo)

```bash
curl -o /home/model-server/examples/image_classifier/densenet161-8d451a50.pth https://download.pytorch.org/models/densenet161-8d451a50.pth
```

4. Now Execute torch-model-archiver command e.g.

```bash
torch-model-archiver --model-name densenet161 --version 1.0 --model-file /home/model-server/examples/image_classifier/densenet_161/model.py --serialized-file /home/model-server/examples/image_classifier/densenet161-8d451a50.pth --export-path /home/model-server/model-store --extra-files /home/model-server/examples/image_classifier/index_to_name.json --handler image_classifier
```

Refer [torch-model-archiver](../model-archiver/README.md) for details.

5. desnet161.mar file should be present at /home/model-server/model-store

# Running TorchServe in a Production Docker Environment.

You may want to consider the following aspects / docker options when deploying torchserve in Production with Docker.

* Shared Memory Size
  * ```shm-size``` - The shm-size parameter allows you to specify the shared memory that a container can use. It enables memory-intensive containers to run faster by giving more access to allocated memory.
* User Limits for System Resources
  * ```--ulimit memlock=-1``` : Maximum locked-in-memory address space.
  * ```--ulimit stack``` : Linux stack size

  The current ulimit values can be viewed by executing ```ulimit -a```. A more exhaustive set of options for resource constraining can be found in the Docker Documentation [here](https://docs.docker.com/config/containers/resource_constraints/), [here](https://docs.docker.com/engine/reference/commandline/run/#set-ulimits-in-container---ulimit) and [here](https://docs.docker.com/engine/reference/run/#runtime-constraints-on-resources)
* Exposing specific ports / volumes between the host & docker env.

    *  ```-p8080:8080 -p8081:8081 -p 8082:8082 -p 7070:7070 -p 7071:7071 ``` 
       TorchServe uses default ports 8080 / 8081 / 8082 for REST based inference, management & metrics APIs and 7070 / 7071 for gRPC APIs. You may want to expose these ports to the host for HTTP & gRPC Requests between Docker & Host.
    * The model store is passed to torchserve with the --model-store option. You may want to consider using a shared volume if you prefer pre populating models in model-store directory.

For example,

```
docker run --rm --shm-size=1g \
        --ulimit memlock=-1 \
        --ulimit stack=67108864 \
        -p8080:8080 \
        -p8081:8081 \
        -p8082:8082 \
        -p7070:7070 \
        -p7071:7071 \
        --mount type=bind,source=/path/to/model/store,target=/tmp/models <container> torchserve --model-store=/tmp/models 
```
