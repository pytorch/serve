# TorchServe on Windows Subsystem for Linux (WSL)
* Ubuntu 18.0.4

## Contents of this Document

* [Setup Ubuntu 18.0.4 on WSL](#setup-ubuntu-1804-on-wsl)
* [Install from binaries](#install-from-binaries)
* [Install from source](#install-from-source)
* [Known limitations of NVIDIA CUDA support on GPU](https://docs.nvidia.com/cuda/wsl-user-guide/index.html#known-limitations)

## Setup Ubuntu 18.0.4 on WSL:

 - [Windows Subsystem for Linux Installation Guide for Windows 10](https://docs.microsoft.com/en-us/windows/wsl/install-win10)
 - [Windows Subsystem for Linux Installation Guide for Windows Server 2019](https://docs.microsoft.com/en-us/windows/wsl/install-on-server)


## Install from binaries

1. Setup Ubuntu Environment

```bash
wget -O - https://raw.githubusercontent.com/pytorch/serve/master/ts_scripts/setup_wsl_ubuntu | bash
echo 'export PATH=$HOME/.local/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```

1. Install JDK 17

```bash
sudo apt-get install openjdk-17-jdk
```

1. Install Dependencies

```
pip install torch torchtext torchvision sentencepiece psutil
pip install torchserve torch-model-archiver
```


## Install from source

1. Clone and install TorchServe

```
git clone https://github.com/pytorch/serve.git
cd serve

./ts_scripts/setup_wsl_ubuntu
export PATH=$HOME/.local/bin:$PATH
python ./ts_scripts/install_from_src.py
echo 'export PATH=$HOME/.local/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```
