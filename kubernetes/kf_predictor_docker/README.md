## Contents of this Document

* [Prerequisites](#prerequisites)
* [Create TorchServe docker image](#create-torchserve-docker-image)
* [Create TorchServe docker image from source](#create-torchserve-docker-image-from-source)
* [Create torch-model-archiver from container](#create-torch-model-archiver-from-container)
* [Running TorchServe docker image in production](#running-torchserve-in-a-production-docker-environment)
* [Create a Torchserve docker image for KFServing]

# Prerequisites

* docker - Refer to the [official docker installation guide](https://docs.docker.com/install/)
* git    - Refer to the [official git set-up guide](https://help.github.com/en/github/getting-started-with-github/set-up-git)
* For base Ubuntu with GPU, install following nvidia container toolkit and driver- 
  * [Nvidia container toolkit](https://github.com/NVIDIA/nvidia-docker#ubuntu-160418042004-debian-jessiestretchbuster)
  * [Nvidia driver](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/install-nvidia-driver.html)

#### Accessing TorchServe APIs inside container

The TorchServe's inference and management APIs can be accessed on localhost over 8080 and 8081 ports respectively. Example :

```bash
curl http://localhost:8080/ping
```

#Create a Docker Image for running Torchserve in KFServing
We have created a docker file by the name Dockerfile_kf.dev to create a docker dev image to run the Torchserve models inside of KFServing. The config_kf.properties should expose the Inference Address to port 8085 since KFServing containers are exposed to port 8080. 


The command to create the docker image is as below:
'''bash

DOCKER_BUILDKIT=1 docker build --file Dockerfile_kf.dev -t <imagename>:<version> .

'''
