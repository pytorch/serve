# TorchServe

TorchServe is a flexible and easy to use tool for serving PyTorch models.

**For full documentation, see [Model Server for PyTorch Documentation](docs/README.md).**

## TorchServe Architecture
![Architecture Diagram](https://user-images.githubusercontent.com/880376/83180095-c44cc600-a0d7-11ea-97c1-23abb4cdbe4d.jpg)

### Terminology:
* **Frontend**: The request/response handling component of TorchServe. This portion of the serving component handles both request/response coming from clients as well manages the models lifecycle.
* **Model Workers**: These workers are responsible for running the actual inference on the models. These are actual running instances of the models.
* **Model**: Models could be a `script_module` (JIT saved models) or `eager_mode_models`. These models can provide custom pre- and post-processing of data along with any other model artifacts such as state_dicts. Models can be loaded from cloud storage or from local hosts.
* **Plugins**: These are custom endpoints or authz/authn or batching algorithms that can be dropped into TorchServe at startup time.
* **Model Store**: This is a directory in which all the loadable models exist.

## Contents of this Document

* [Install TorchServe](#install-torchserve)
* [Serve a Model](#serve-a-model)
* [Quick start with docker](#quick-start-with-docker)
* [Contributing](#contributing)

## Install TorchServe

Conda instructions are provided in more detail, but you may also use `pip` and `virtualenv` if that is your preference.
**Note:** Java 11 is required. Instructions for installing Java 11 for Ubuntu or macOS are provided in the [Install with Conda](#install-with-conda) section.

### Install with pip

1. Install Java 11

    ```bash
    sudo apt-get install openjdk-11-jdk
    ```

1. Use `pip` to install TorchServe and the model archiver:

    ``` bash
    pip install torch torchtext torchvision sentencepiece psutil future
    pip install torchserve torch-model-archiver
    ```

### Install with Conda
**Note:** For Conda, Python 3.8 is required to run Torchserve

#### Ubuntu

1. Install Java 11

    ```bash
    sudo apt-get install openjdk-11-jdk
    ```

1. Install Conda (https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html)
1. Create an environment and install torchserve and torch-model-archiver
    For CPU

    ```bash
    conda create --name torchserve torchserve torch-model-archiver psutil future pytorch torchtext torchvision -c pytorch -c powerai
    ```

    For GPU

    ```bash
    conda create --name torchserve torchserve torch-model-archiver psutil future pytorch torchtext torchvision cudatoolkit=10.1 -c pytorch -c powerai
    ```

1. Activate the environment

    ```bash
    source activate torchserve
    ```

2. Optional if using torchtext models
    ```bash
    pip install sentencepiece
    ```

#### macOS

1. Install Java 11

    ```bash
    brew tap AdoptOpenJDK/openjdk
    brew cask install adoptopenjdk11
    ```

1. Install Conda (https://docs.conda.io/projects/conda/en/latest/user-guide/install/macos.html)
1. Create an environment and install torchserve and torch-model-archiver

    ```bash
    conda create --name torchserve torchserve torch-model-archiver psutil future pytorch torchtext torchvision -c pytorch -c powerai
    ```

1. Activate the environment

    ```bash
    source activate torchserve
    ```

2. Optional if using torchtext models
    ```bash
    pip install sentencepiece
    ```

Now you are ready to [package and serve models with TorchServe](#serve-a-model).

### Install TorchServe for development

If you plan to develop with TorchServe and change some of the source code, you must install it from source code.

Please deactivate any conda env that you might be within.
Run the following script from the top of the source directory.

NOTE: This script uninstalls existing `torchserve` and `torch-model-archiver` installations

#### For Debian Based Systems
Verified on EC2 instances running Ubuntu DL AMI 28.x

```
./scripts/install_from_src_ubuntu
```
#### For MAC OS

```
./scripts/install_from_src_macos
```

For information about the model archiver, see [detailed documentation](model-archiver/README.md).

## Serve a model

This section shows a simple example of serving a model with TorchServe. To complete this example, you must have already [installed TorchServe and the model archiver](#install-with-pip).

To run this example, clone the TorchServe repository:

```bash
git clone https://github.com/pytorch/serve.git
```

Then run the following steps from the parent directory of the root of the repository.
For example, if you cloned the repository into `/home/my_path/serve`, run the steps from `/home/my_path`.

### Store a Model

To serve a model with TorchServe, first archive the model as a MAR file. You can use the model archiver to package a model.
You can also create model stores to store your archived models.

1. Create a directory to store your models.

    ```bash
    mkdir model_store
    ```

1. Download a trained model.

    ```bash
    wget https://download.pytorch.org/models/densenet161-8d451a50.pth
    ```

1. Archive the model by using the model archiver. The `extra-files` param uses fa file from the `TorchServe` repo, so update the path if necessary.

    ```bash
    torch-model-archiver --model-name densenet161 --version 1.0 --model-file ./serve/examples/image_classifier/densenet_161/model.py --serialized-file densenet161-8d451a50.pth --export-path model_store --extra-files ./serve/examples/image_classifier/index_to_name.json --handler image_classifier
    ```

For more information about the model archiver, see [Torch Model archiver for TorchServe](model-archiver/README.md)

### Start TorchServe to serve the model

After you archive and store the model, use the `torchserve` command to serve the model.

```bash
torchserve --start --ncs --model-store model_store --models densenet161.mar
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
curl http://127.0.0.1:8080/predictions/densenet161 -T kitten.jpg
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


### Concurrency And Number of Workers
TorchServe exposes configurations which allows the user to configure the number of worker threads on CPU and GPUs. This is an important config property that can speed up the server depending on the workload.
*Note: the following property has bigger impact under heavy workloads.*
If TorchServe is hosted on a machine with GPUs, there is a config property called `number_of_gpu` which tells the server to use a specific number of GPU per model. In cases where we register multiple models with the server, this will apply to all the models registered. If this is set to a low value (ex: 0 or 1), it will result in under-utilization of GPUs. On the contrary, setting to a high value (>= max GPUs available on the system) results in as many workers getting spawned per model. Clearly, this will result in unnecessary contention for GPUs and can result in sub-optimal scheduling of threads to GPU.
```
ValueToSet = (Number of Hardware GPUs) / (Number of Unique Models)
```


## Quick Start with Docker
Refer [torchserve docker](docker/README.md) for details.

## Learn More

* [Full documentation on TorchServe](docs/README.md)
* [Manage models API](docs/management_api.md)
* [Inference API](docs/inference_api.md)
* [Package models for use with TorchServe](model-archiver/README.md)
* [TorchServe model zoo for pre-trained and pre-packaged models-archives](docs/model_zoo.md)

## Contributing

We welcome all contributions!

To learn more about how to contribute, see the contributor guide [here](https://github.com/pytorch/serve/blob/master/CONTRIBUTING.md).

To file a bug or request a feature, please file a GitHub issue. For filing pull requests, please use the template [here](https://github.com/pytorch/serve/blob/master/pull_request_template.md). Cheers!


*TorchServe acknowledges the [Multi Model Server (MMS)](https://github.com/awslabs/multi-model-server) project from which it was derived*
