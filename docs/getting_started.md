# Getting started

## Install TorchServe and torch-model-archiver

1. Install dependencies

    Note: For Conda, Python >=3.8 is required to run Torchserve.

    #### For Debian Based Systems/ MacOS

     - For CPU

        ```bash
        python ./ts_scripts/install_dependencies.py
        ```

     - For GPU with Cuda 10.2. Options are `cu92`, `cu101`, `cu102`, `cu111`, `cu113`, `cu116`, `cu117`, `cu118`

       ```bash
       python ./ts_scripts/install_dependencies.py --cuda=cu102
       ```

     Note: PyTorch 1.9+ will not support cu92 and cu101. So TorchServe only supports cu92 and cu101 up to PyTorch 1.8.1.

    #### For Windows

    Refer to the documentation [here](./torchserve_on_win_native.md).

2. Install torchserve, torch-model-archiver and torch-workflow-archiver

    For [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install)
    Note: Conda packages are not supported for Windows. Refer to the documentation [here](./torchserve_on_win_native.md).
    ```
    conda install torchserve torch-model-archiver torch-workflow-archiver -c pytorch
    ```

    For Pip
    ```
    pip install torchserve torch-model-archiver torch-workflow-archiver
    ```

Now you are ready to [package and serve models with TorchServe](#serve-a-model).

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

For more information about the model archiver, see [Torch Model archiver for TorchServe](https://github.com/pytorch/serve/tree/master/model-archiver/README.md)

### Start TorchServe to serve the model

After you archive and store the model, use the `torchserve` command to serve the model.

```bash
torchserve --start --ncs --model-store model_store --models densenet161.mar
```

After you execute the `torchserve` command above, TorchServe runs on your host, listening for inference requests.

**Note**: If you specify model(s) when you run TorchServe, it automatically scales backend workers to the number equal to available vCPUs (if you run on a CPU instance) or to the number of available GPUs (if you run on a GPU instance). In case of powerful hosts with a lot of compute resources (vCPUs or GPUs), this start up and autoscaling process might take considerable time. If you want to minimize TorchServe start up time you should avoid registering and scaling the model during start up time and move that to a later point by using corresponding [Management API](./management_api.md#register-a-model), which allows finer grain control of the resources that are allocated for any particular model).

### Get predictions from a model

To test the model server, send a request to the server's `predictions` API. TorchServe supports all [inference](./inference_api.md) and [management](./management_api.md) apis through both [gRPC](./grpc_api.md) and [HTTP/REST](./rest_api.md).

#### Using GRPC APIs through python client

 - Install grpc python dependencies :

```bash
pip install -U grpcio protobuf grpcio-tools
```

 - Generate inference client using proto files

```bash
python -m grpc_tools.protoc --proto_path=frontend/server/src/main/resources/proto/ --python_out=ts_scripts --grpc_python_out=ts_scripts frontend/server/src/main/resources/proto/inference.proto frontend/server/src/main/resources/proto/management.proto
```

 - Run inference using a sample client [gRPC python client](https://github.com/pytorch/serve/blob/master/ts_scripts/torchserve_grpc_client.py)

```bash
python ts_scripts/torchserve_grpc_client.py infer densenet161 examples/image_classifier/kitten.jpg
```

#### Using REST APIs

As an example we'll download the below cute kitten with

![kitten](images/kitten_small.jpg)

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

Now you've seen how easy it can be to serve a deep learning model with TorchServe! [Would you like to know more?](./server.md)

### Stop TorchServe

To stop the currently running TorchServe instance, run:

```bash
torchserve --stop
```

### Inspect the logs
All the logs you've seen as output to stdout related to model registration, management, inference are recorded in the `/logs` folder.

High level performance data like Throughput or Percentile Precision can be generated with [Benchmark](https://github.com/pytorch/serve/tree/master/benchmarks/README.md) and visualized in a report.

### Contributing

If you plan to develop with TorchServe and change some source code, follow the [contributing guide](https://github.com/pytorch/serve/blob/master/CONTRIBUTING.md).
