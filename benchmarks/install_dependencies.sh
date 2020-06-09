#!/bin/bash

# This file contains the installation setup for running benchmarks on EC2 isntance.
# To run on a machine with GPU : ./install_dependencies.sh True
# To run on a machine with CPU : ./install_dependencies.sh False

set -ex

sudo apt-get update
sudo apt-get -y upgrade
echo "Setting up your Ubuntu machine to load test TS"
sudo apt-get install -y \
        python3-dev \
        python-psutil \
        python3-pip \
        openjdk-11-jdk \
        g++ \
        curl \
        vim \
        git \
        file \
        build-essential \
        && cd /tmp \
        && curl -O https://bootstrap.pypa.io/get-pip.py \
        && python3 get-pip.py

/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install.sh)"
export PATH="/home/linuxbrew/.linuxbrew/Homebrew/bin/:$PATH"
sudo update-alternatives --install /usr/bin/python python /usr/bin/python3 1
sudo update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

if [[ $1 = True ]]
then
        echo "Installing pip packages for GPU"
        sudo apt install -y nvidia-cuda-toolkit
        pip install future psutil pillow --user
        pip install torch==1.4.0+cu92 torchvision==0.5.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html
else
        echo "Installing pip packages for CPU"
        pip install future psutil pillow torch torchvision --user

fi

pip install pandas

echo "Installing JMeter through Brew"
# Script would end on errors, but everything works fine
{
    yes '' | brew update
} || {
    true
}
{
    brew install jmeter
} || {
    true
}

CELLAR="/home/linuxbrew/.linuxbrew/Homebrew/Cellar/jmeter/"

if [ $(ls -1d $CELLAR/* | wc -l) -gt 1 ];then
  echo "Multiple versions of JMeter installed. Exiting..."
  exit 1
fi

JMETER_HOME=`find $CELLAR ! -path $CELLAR -type d -maxdepth 1`

wget https://jmeter-plugins.org/get/ -O $JMETER_HOME/libexec/lib/ext/jmeter-plugins-manager-1.3.jar
wget http://search.maven.org/remotecontent?filepath=kg/apc/cmdrunner/2.2/cmdrunner-2.2.jar -O $JMETER_HOME/libexec/lib/cmdrunner-2.2.jar
java -cp $JMETER_HOME/libexec/lib/ext/jmeter-plugins-manager-1.3.jar org.jmeterplugins.repository.PluginManagerCMDInstaller
$JMETER_HOME/libexec/bin/PluginsManagerCMD.sh install jpgc-synthesis=2.1,jpgc-filterresults=2.1,jpgc-mergeresults=2.1,jpgc-cmd=2.1,jpgc-perfmon=2.1

echo "Install docker"
sudo apt-get remove -y docker docker-engine docker.io
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
# shellcheck disable=SC1073
{
    sudo groupadd docker ||  true
} || {
    true
}
{
    sudo gpasswd -a $USER docker
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
    docker run --gpus all --rm nvidia/cuda nvidia-smi
fi
