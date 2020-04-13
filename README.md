# TorchServe

TorchServe is a flexible and easy to use tool for serving PyTorch models.

For full documentation, see [Model Server for PyTorch Documentation](docs/README.md).

## Contents of this Document

* [Install TorchServe for local environment](#install-torchserve-for-local-environment)
* [Serve a Model](#serve-a-model)
* [Quick start with docker](#quick-start-with-docker)
* [Contributing](#contributing)

## Install TorchServe for local environment

### Prerequisites

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
    sudo yum install java-11-openjdk
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

## Serve a model

This section shows a simple example of serving a model with TorchServe. To complete this example, you must have already installed TorchServe and the model archiver. 

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

After you execute the `torchserve` command above, TorchServe runs on your host, listening for inference requests.

**Note**: If you specify model(s) when you run TorchServe, it automatically scales backend workers to the number equal to available vCPUs (if you run on a CPU instance) or to the number of available GPUs (if you run on a GPU instance). In case of powerful hosts with a lot of compute resoures (vCPUs or GPUs). This start up and autoscaling process might take considerable time. If you want to minimize TorchServe start up time you avoid registering and scaling the model during start up time and move that to a later point by using corresponding [Management API](docs/management_api.md#register-a-model), which allows finer grain control of the resources that are allocated for any particular model).

### Get predictions from a model

To test the model server, send a request to the server's `predictions` API.

Complete the following steps:

* Open a new terminal window (other than the one running TorchServe).
* Use `curl` to download one of these [cute pictures of a kitten](https://www.google.com/search?q=cute+kitten&tbm=isch&hl=en&cr=&safe=images)
  and use the  `-o` flag to name it `kitten.jpg` for you.
* Use `curl` to send `POST` to the TorchServe `predict` endpoint with the kitten's image.

![kitten](docs/images/kitten_small.jpg)

The following code completes all three steps:

```bash
curl -O https://s3.amazonaws.com/model-server/inputs/kitten.jpg
curl -X POST http://127.0.0.1:8080/predictions/densenet161 -T kitten.jpg
```

The predict endpoint returns a prediction response in JSON. It will look something like the following result:

```json
[
  {
    "tiger_cat": 0.46933549642562866
  },
  {
    "tabby": 0.4633878469467163
  },
  {
    "Egyptian_cat": 0.06456148624420166
  },
  {
    "lynx": 0.0012828214094042778
  },
  {
    "plastic_bag": 0.00023323034110944718
  }
]
```

You will see this result in the response to your `curl` call to the predict endpoint, and in the server logs in the terminal window running TorchServe. It's also being [logged locally with metrics](docs/metrics.md).

Now you've seen how easy it can be to serve a deep learning model with TorchServe! [Would you like to know more?](docs/server.md)

### Stop the running TorchServe

To stop the currently running TorchServe instance, run the following command:

```bash
torchserve --stop
```

You see output specifying that TorchServe has stopped.

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
