#!/bin/bash

# This file contains the installation setup for running benchmarks on EC2 isntance.

# To run on a machine with GPU : ./install_dependencies True

# To run on a machine with CPU : ./install_dependencies False

set -ex

sudo apt-get update

sudo apt-get -y upgrade

echo "Setting up your Ubuntu machine to load test MMS"

sudo apt-get install -y \

python \

python-pip \

python3-pip \

python3-tk \

python-psutil \

default-jre \

default-jdk \

build-essential

if [[ $1 = True ]]

then

echo "Installing pip packages for GPU"

sudo apt install -y nvidia-cuda-toolkit

pip install future psutil mxnet-cu92 pillow --user

else

echo "Installing pip packages for CPU"

pip install future psutil mxnet pillow --user

fi

pip3 install pandas

echo "Install docker"

sudo apt-get remove docker docker-engine docker.io

sudo apt-get install -y \

apt-transport-https \

ca-certificates \

curl \

software-properties-common

curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -

sudo add-apt-repository \

"deb [arch=amd64] https://download.docker.com/linux/ubuntu \

$(lsb_release -cs) \

stable"

sudo apt-get update

sudo apt-get install -y docker-ce

{

sudo groupadd docker || {true}

} || {

true

}

{

gpasswd -a $USER docker

} || {

true

}

if [[ $1 = True ]]

then

echo "Installing nvidia-docker"

# If you have nvidia-docker 1.0 installed: we need to remove it and all existing GPU containers

{

docker volume ls -q -f driver=nvidia-docker | xargs -r -I{} -n1 docker ps -q -a -f volume={} | xargs -r docker rm -f

} || {

true

}

{

sudo apt-get purge -y nvidia-docker

} || {

true

}

# Add the package repositories

curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | \

sudo apt-key add -

distribution=$(. /etc/os-release;echo $ID$VERSION_ID)

curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \

sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update

# Install nvidia-docker2 and reload the Docker daemon configuration

sudo apt-get install -y nvidia-docker2

sudo pkill -SIGHUP dockerd

# Test nvidia-smi with the latest official CUDA image

docker run --runtime=nvidia --rm nvidia/cuda nvidia-smi

fi

