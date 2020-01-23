#!/bin/bash

# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#     http://www.apache.org/licenses/LICENSE-2.0
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

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
        linuxbrew-wrapper \
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

echo "Installing JMeter through Brew"
# Script would end on errors, but everything works fine
{
    yes '' | brew update
} || {
    true
}
{
    brew install jmeter --with-plugins
} || {
    true
}

wget https://jmeter-plugins.org/get/ -O /home/ubuntu/.linuxbrew/Cellar/jmeter/5.0/libexec/lib/ext/jmeter-plugins-manager-1.3.jar
wget http://search.maven.org/remotecontent?filepath=kg/apc/cmdrunner/2.2/cmdrunner-2.2.jar -O /home/ubuntu/.linuxbrew/Cellar/jmeter/5.0/libexec/lib/cmdrunner-2.2.jar
java -cp /home/ubuntu/.linuxbrew/Cellar/jmeter/5.0/libexec/lib/ext/jmeter-plugins-manager-1.3.jar org.jmeterplugins.repository.PluginManagerCMDInstaller
/home/ubuntu/.linuxbrew/Cellar/jmeter/5.0/libexec/bin/PluginsManagerCMD.sh install jpgc-synthesis=2.1,jpgc-filterresults=2.1,jpgc-mergeresults=2.1,jpgc-cmd=2.1,jpgc-perfmon=2.1

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
