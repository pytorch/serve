# TorchServe

TorchServe is a flexible and easy to use tool for serving PyTorch models.

For full documentation, see [Model Server for PyTorch Documentation](docs/README.md).

## Contents of this Document

* [Install TorchServe](#install-torchserve)
* [Prerequisites](#prerequisites)
* [Python virtual environment setup](#python-virtual-environment-setup)
* [Quick Start with docker](#quick-start-with-docker)
* [Quick Start for local environment](#quick-start-guide-for-local-environment)
* [Serve a Model](#serve-a-model)
* [Other Features](#other-features)
* [Contributing](#contributing)

## Install TorchServe

## Prerequisites

Before proceeding further with this document, make sure you have the following prerequisites.

1. Ubuntu, CentOS, or macOS. Windows support is experimental. The following instructions will focus on Linux and macOS only.
1. Python     - TorchServe requires python to run the workers.
1. pip        - Pip is a python package management system.
1. Java 11    - TorchServe requires Java 11 to start. You have the following options for installing Java 11:

    For Ubuntu:

    ```bash
    sudo apt-get install openjdk-11-jdk
    ```

    For CentOS:

    ```bash
    openjdk-11-jdk
    sudo yum install java-11-openjdk
    ```

    For macOS

    ```bash
    brew tap AdoptOpenJDK/openjdk
    brew cask install adoptopenjdk11
    ```

## Python virtual environment setup

We recommend installing and running TorchServe in a virtual environment. It's a good practice to run and install all of the Python dependencies in virtual environments. This will provide isolation of the dependencies and ease dependency management.

* **Use Virtualenv** : This is used to create virtual Python environments. You may install and activate a virtualenv for Python 3.7 as follows:

```bash
pip install virtualenv
```

Then create a virtual environment:

```bash
# Assuming we want to run python3.7 in /usr/local/bin/python3.7
virtualenv -p /usr/local/bin/python3.7 /tmp/pyenv3
# Enter this virtual environment as follows
source /tmp/pyenv3/bin/activate
```

Refer to the [Virtualenv documentation](https://virtualenv.pypa.io/en/stable/) for further information.

* **Use Anaconda** : This is package, dependency and environment manager. You may download and install Anaconda as follows :
[Download anaconda distribution](https://www.anaconda.com/distribution/#download-section)

Then create a virtual environment using conda.

```bash
conda create -n myenv
source activate myenv
```

#### Install torch

TorchServe won't install the PyTorch engine by default. If it isn't already installed in your virtual environment, you must install the PyTorch pip packages.

* For virtualenv

```bash
#For CPU/GPU
pip install torch torchvision torchtext
```

* For conda

The `torchtext` package has a dependency on `sentencepiece`, which is not available via Anaconda. You can install it via `pip`:

```bash
pip install sentencepiece
```

```bash
#For CPU
conda install psutil pytorch torchvision torchtext -c pytorch
```

```bash
#For GPU
conda install future psutil pytorch torchvision cudatoolkit=10.1 torchtext -c pytorch
```

## Quick Start with docker

### Start TorchServe using docker image

#### Docker specific prerequisites

* docker - Refer [official docker installation guide](https://docs.docker.com/install/)
* git    - Refer [official git set-up guide](https://help.github.com/en/github/getting-started-with-github/set-up-git)

#### Building docker image

```bash
git clone https://github.com/pytorch/serve.git
cd serve
./build_image.sh
```

The above command builds the TorchServe image for CPU device with `master` branch

To create image for specific branch use following command :
```bash
./build_image.sh -b <branch_name>
```

To create image for GPU device use following command :
```bash
./build_image.sh --gpu
```

To create image for GPU device with specific branch use following command :
```bash
./build_image.sh -b <branch_name> --gpu
```

**Running docker image and starting TorchServe inside container with pre-registered resnet-18 image classification model**

```bash
./start.sh
```

**For pre-trained and pre-packaged models-archives refer [TorchServe model zoo](docs/model_zoo.md)**
**For managing models with TorchServe refer [management api documentation](docs/management_api.md)**
**For running inference on registered models with TorchServe refer [inference api documentation](docs/inference_api.md)**

## Quick Start for local environment

### Installing TorchServe with pip

#### Setup

**Step 1:** Install TorchServe as follows:

```bash
git clone https://github.com/pytorch/serve.git
cd serve
pip install .
```

**Notes:**

* If `pip install .`  fails, run `python setup.py install` and install the following python packages using `pip install` : Pillow, psutil, future
* See the [advanced installation](docs/install.md) page for more options and troubleshooting.

### Install torch-model-archiver

* Install torch-model-archiver as follows:

```bash
cd serve/model-archiver
pip install .
```

For information about the model archiver, see [detailed documentation](model-archiver/README.md).

### Install TorchServe for development

If you plan to develop with TorchServe and change some of the source code, install it from source code and make your changes executable with this command:

```bash
pip install -e .
```

To upgrade TorchServe from source code and make changes executable, run:

```bash
pip install -U -e .
```

## Troubleshoot Installation

Here is an easy example for serving an object classification model (make sure to run it at the root of the repository):

```bash
wget https://download.pytorch.org/models/densenet161-8d451a50.pth
torch-model-archiver --model-name densenet161 --version 1.0 --model-file examples/image_classifier/densenet_161/model.py --serialized-file densenet161-8d451a50.pth --extra-files examples/image_classifier/index_to_name.json --handler image_classifier
mkdir model_store
mv densenet161.mar model_store/
torchserve --start --model-store model_store --models densenet161=densenet161.mar
```

## Contributing

We welcome all contributions!

To file a bug or request a feature, please file a GitHub issue. Pull requests are welcome.

## Experimental Release Roadmap

Below, in order, is a prioritized list of tasks for this repository.

### v0.1 Plan

* [ ] CI (initially AWS CodeBuild)
* [x] Default handler
  * [x] Handle eager-mode and TorchScript (tracing and scripting)
  * [x] Add zero-code pre and post-processing for Image Classification
* [x] Basic examples
  * [x] Eager-mode image classifier
    * [x] TorchScript image classifier
    * [x] Custom neural network
* [x] Basic docs (install, serve a model and use it for inference)

### v0.2 Plan

* [ ] Basic unit tests
* [ ] Versioning
