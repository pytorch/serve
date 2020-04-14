# TorchServe

TorchServe is a flexible and easy to use tool for serving PyTorch models.

For full documentation, see [Model Server for PyTorch Documentation](docs/README.md).

## Contents of this Document

* [Install TorchServe](#install-torchserve)
* [Quick Start with docker](#quick-start-with-docker)
* [Quick Start for local environment](#quick-start-for-local-environment)
* [Serve a Model](#serve-a-model)
* [Contributing](#contributing)

## Install TorchServe

## Quick Start with docker

### Start TorchServe using docker image

#### Prerequisites

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

**For managing models with TorchServe refer [management api documentation](docs/management_api.md)**
**For running inference on registered models with TorchServe refer [inference api documentation](docs/inference_api.md)**

## Quick Start for local environment

### Prerequisites

Before proceeding further with this document, make sure you have the following prerequisites.

1. Ubuntu or macOS. The following instructions will focus on Linux and macOS only.
1. Python     - TorchServe requires python to run the workers.
1. pip        - Pip is a python package management system.
1. Java 11    - TorchServe requires Java 11 to start. You have the following options for installing Java 11:

    For Ubuntu:

    ```bash
    sudo apt-get install openjdk-11-jdk
    ```

    For macOS

    ```bash
    brew tap AdoptOpenJDK/openjdk
    brew cask install adoptopenjdk11
    ```

### Install TorchServe with pip

#### Setup

**Step 1:** Setup a Virtual Environment

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

**Step 2:** Install torch

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

**Step 3:** Install TorchServe as follows:

```bash
git clone https://github.com/pytorch/serve.git
cd serve
pip install .
```

**Notes:**

* If `pip install .`  fails, install the following python packages using `pip install` : Pillow, psutil, future and run `python setup.py install`.

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

## Serve a model

To run this example, clone the TorchServe repository and navigate to the root of the repository:

```bash
git clone https://github.com/pytorch/serve.git
cd serve
```

Then run the following steps from the root of the repository.

### Store a Model

To serve a model with TorchServe, first archive the model as a MAR file. You can use the model archiver to package a model.
You can also create model stores to store your archived models.

The following code gets a trained model, archives the model by using the model archiver, and then stores the model in a model store.

```bash
wget https://download.pytorch.org/models/densenet161-8d451a50.pth
torch-model-archiver --model-name densenet161 --version 1.0 --model-file examples/image_classifier/densenet_161/model.py --serialized-file densenet161-8d451a50.pth --extra-files examples/image_classifier/index_to_name.json --handler image_classifier
mkdir model_store
mv densenet161.mar model_store/
```

For more information about the model archiver, see [Torch Model archiver for TorchServe](../model-archiver/README.md)

### Start TorchServe to serve the model

After you archive and store the model, use the `torchserve` command to serve the model.

```bash
torchserve --start --model-store model_store --models densenet161=densenet161.mar
```

For more details refer [quick start guide for model serving](docs/quick_start.md)

## Contributing

We welcome all contributions!

To file a bug or request a feature, please file a GitHub issue. Pull requests are welcome.

## Experimental Release Roadmap

Below, in order, is a prioritized list of tasks for this repository.

### v0.1 Plan

* [x] CI (initially AWS CodeBuild)
* [x] Default handler
  * [x] Handle eager-mode and TorchScript (tracing and scripting)
  * [x] Add zero-code pre and post-processing for Image Classification
  * [x] Add zero-code pre and post-processing for Text Classification
  * [x] Add zero-code pre and post-processing for Image Segmentation
  * [x] Add zero-code pre and post-processing for Object Detection
* [x] Basic examples
  * [x] Eager-mode image classifier
  * [x] TorchScript image classifier
  * [x] Custom neural network
* [x] Basic docs (install, serve a model and use it for inference)
* [x] Basic unit tests
* [x] Versioning
* [x] Snapshot
