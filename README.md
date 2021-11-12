# TorchServe

TorchServe is a flexible and easy to use tool for serving PyTorch models.

**For full documentation, see [Model Server for PyTorch Documentation](docs/README.md).**

## TorchServe Architecture
![Architecture Diagram](https://user-images.githubusercontent.com/880376/83180095-c44cc600-a0d7-11ea-97c1-23abb4cdbe4d.jpg)

### Terminology:
* **Frontend**: The request/response handling component of TorchServe. This portion of the serving component handles both request/response coming from clients and manages the lifecycles of the models.
* **Model Workers**: These workers are responsible for running the actual inference on the models.
* **Model**: Models could be a `script_module` (JIT saved models) or `eager_mode_models`. These models can provide custom pre- and post-processing of data along with any other model artifacts such as state_dicts. Models can be loaded from cloud storage or from local hosts.
* **Plugins**: These are custom endpoints or authz/authn or batching algorithms that can be dropped into TorchServe at startup time.
* **Model Store**: This is a directory in which all the loadable models exist.

## Contents of this Document

* [Install TorchServe](#install-torchserve-and-torch-model-archiver)
* [Install TorchServe on Windows](docs/torchserve_on_win_native.md)
* [Install TorchServe on Windows Subsystem for Linux](docs/torchserve_on_wsl.md)
* [Serve a Model](#serve-a-model)
* [Serve a Workflow](docs/workflows.md)
* [Quick start with docker](#quick-start-with-docker)
* [Highlighted Examples](#highlighted-examples)
* [Featured Community Projects](#featured-community-projects)
* [Contributing](#contributing)

## Install TorchServe and torch-model-archiver

1. Install dependencies

    Note: For Conda, Python 3.8 is required to run Torchserve.

    #### For Debian Based Systems/ MacOS
    
     - For CPU

        ```bash
        python ./ts_scripts/install_dependencies.py
        ```
        
     - For GPU with Cuda 10.2. Options are `cu92`, `cu101`, `cu102`, `cu111`, `cu113`

       ```bash
       python ./ts_scripts/install_dependencies.py --cuda=cu102
       ```
       
     Note: PyTorch 1.9+ will not support cu92 and cu101. So TorchServe only supports cu92 and cu101 up to PyTorch 1.8.1.  

    #### For Windows

    Refer to the documentation [here](docs/torchserve_on_win_native.md).

2. Install torchserve, torch-model-archiver and torch-workflow-archiver

    For [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install)  
    Note: Conda packages are not supported for Windows. Refer to the documentation [here](docs/torchserve_on_win_native.md).
    ```
    conda install torchserve torch-model-archiver torch-workflow-archiver -c pytorch
    ```
   
    For Pip
    ```
    pip install torchserve torch-model-archiver torch-workflow-archiver
    ```

Now you are ready to [package and serve models with TorchServe](#serve-a-model).

### Install TorchServe for development

If you plan to develop with TorchServe and change some source code, you must install it from source code.

Ensure that you have `python3` installed, and the user has access to the site-packages or `~/.local/bin` is added to the `PATH` environment variable.

Run the following script from the top of the source directory.

NOTE: This script uninstalls existing `torchserve`, `torch-model-archiver` and `torch-workflow-archiver` installations

#### For Debian Based Systems/ MacOS

```
python ./ts_scripts/install_dependencies.py --environment=dev
python ./ts_scripts/install_from_src.py
```

Use `--cuda` flag with `install_dependencies.py` for installing cuda version specific dependencies. Possible values are `cu111`, `cu102`, `cu101`, `cu92`

#### For Windows

Refer to the documentation [here](docs/torchserve_on_win_native.md).

For information about the model archiver, see [detailed documentation](model-archiver/README.md).

## Serve a model

This section shows a simple example of serving a model with TorchServe. To complete this example, you must have already [installed TorchServe and the model archiver](#install-torchserve-and-torch-model-archiver).

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

1. Archive the model by using the model archiver. The `extra-files` param uses a file from the `TorchServe` repo, so update the path if necessary.

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

**Note**: If you specify model(s) when you run TorchServe, it automatically scales backend workers to the number equal to available vCPUs (if you run on a CPU instance) or to the number of available GPUs (if you run on a GPU instance). In case of powerful hosts with a lot of compute resources (vCPUs or GPUs), this start up and autoscaling process might take considerable time. If you want to minimize TorchServe start up time you should avoid registering and scaling the model during start up time and move that to a later point by using corresponding [Management API](docs/management_api.md#register-a-model), which allows finer grain control of the resources that are allocated for any particular model).

### Get predictions from a model

To test the model server, send a request to the server's `predictions` API. TorchServe supports all [inference](docs/inference_api.md) and [management](docs/management_api.md) api's through both [gRPC](docs/grpc_api.md) and [HTTP/REST](docs/rest_api.md).

#### Using GRPC APIs through python client

 - Install grpc python dependencies :
 
```bash
pip install -U grpcio protobuf grpcio-tools
```

 - Generate inference client using proto files

```bash
python -m grpc_tools.protoc --proto_path=frontend/server/src/main/resources/proto/ --python_out=ts_scripts --grpc_python_out=ts_scripts frontend/server/src/main/resources/proto/inference.proto frontend/server/src/main/resources/proto/management.proto
```

 - Run inference using a sample client [gRPC python client](ts_scripts/torchserve_grpc_client.py)

```bash
python ts_scripts/torchserve_grpc_client.py infer densenet161 examples/image_classifier/kitten.jpg
```

#### Using REST APIs

As an example we'll download the below cute kitten with

![kitten](docs/images/kitten_small.jpg)

```bash
curl -O https://raw.githubusercontent.com/pytorch/serve/master/docs/images/kitten_small.jpg
```

And then call the prediction endpoint 

```bash
curl http://127.0.0.1:8080/predictions/densenet161 -T kitten_small.jpg
```

Which will return the following JSON object

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

All interactions with the endpoint will be logged in the `logs/` directory, so make sure to check it out!

Now you've seen how easy it can be to serve a deep learning model with TorchServe! [Would you like to know more?](docs/server.md)

### Stop TorchServe

To stop the currently running TorchServe instance, run:

```bash
torchserve --stop
```

### Inspect the logs
All the logs you've seen as output to stdout related to model registration, management, inference are recorded in the `/logs` folder.

High level performance data like Throughput or Percentile Precision can be generated with [Benchmark](benchmark/README.md) and visualized in a report.

### Concurrency And Number of Workers
TorchServe exposes configurations that allow the user to configure the number of worker threads on CPU and GPUs. There is an important config property that can speed up the server depending on the workload.
*Note: the following property has bigger impact under heavy workloads.*

**CPU**: there is a config property called `workers` which sets the number of worker threads for a model. The best value to set `workers` to is to start with `num physical cores / 2` and increase it as much possible after setting `torch.set_num_threads(1)` in your handler.

**GPU**: there is a config property called `number_of_gpu` that tells the server to use a specific number of GPUs per model. In cases where we register multiple models with the server, this will apply to all the models registered. If this is set to a low value (ex: 0 or 1), it will result in under-utilization of GPUs. On the contrary, setting to a high value (>= max GPUs available on the system) results in as many workers getting spawned per model. Clearly, this will result in unnecessary contention for GPUs and can result in sub-optimal scheduling of threads to GPU.
```
ValueToSet = (Number of Hardware GPUs) / (Number of Unique Models)
```

## Quick Start with Docker
Refer to [torchserve docker](docker/README.md) for details.

## Highlighted Examples
* [HuggingFace Transformers](examples/Huggingface_Transformers)
* [MultiModal models with MMF](https://github.com/pytorch/serve/tree/master/examples/MMF-activity-recognition) combining text, audio and video
* Complex workflows, models chained together in a dependency graph
  - [Dog Breed Classification](examples/Workflows/dog_breed_classification)
  - [Neural Machine Translation](examples/Workflows/nmt_tranformers_pipeline)

Feel free to skim the full list of [available examples](examples/README.md)

## Featured Community Projects
* [SynthDet by Unity Technologies](https://github.com/Unity-Technologies/SynthDet)
* [Torchserve dashboard by cceyda](https://github.com/cceyda/torchserve-dashboard)

## Learn More

* [Full documentation on TorchServe](docs/README.md)
* [Model Management API](docs/management_api.md)
* [Inference API](docs/inference_api.md)
* [Metrics API](docs/metrics.md)
* [Package models for use with TorchServe](model-archiver/README.md)
* [Deploying TorchServe with Kubernetes](kubernetes/README.md)
* [TorchServe Workflows](examples/Workflows/README.md)
* [TorchServe model zoo for pre-trained and pre-packaged models-archives](docs/model_zoo.md)

## Contributing

We welcome all contributions!

To learn more about how to contribute, see the contributor guide [here](https://github.com/pytorch/serve/blob/master/CONTRIBUTING.md).

To file a bug or request a feature, please file a GitHub issue. For filing pull requests, please use the template [here](https://github.com/pytorch/serve/blob/master/pull_request_template.md). Cheers!

## Disclaimer 
This repository is jointly operated and maintained by Amazon, Facebook and a number of individual contributors listed in the [CONTRIBUTORS](https://github.com/pytorch/serve/graphs/contributors) file. For questions directed at Facebook, please send an email to opensource@fb.com. For questions directed at Amazon, please send an email to torchserve@amazon.com. For all other questions, please open up an issue in this repository [here](https://github.com/pytorch/serve/issues).

*TorchServe acknowledges the [Multi Model Server (MMS)](https://github.com/awslabs/multi-model-server) project from which it was derived*
