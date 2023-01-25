# Batch Inference with TorchServe

## Contents of this Document

* [Introduction](#introduction)
* [Prerequisites](#prerequisites)
* [Batch Inference with TorchServe's default handlers](#batch-inference-with-torchserves-default-handlers)
* [Batch Inference with TorchServe using ResNet-152 model](#batch-inference-with-torchserve-using-resnet-152-model)  
* [Demo to configure TorchServe ResNet-152 model with batch-supported model](#demo-to-configure-torchserve-resnet-152-model-with-batch-supported-model)
* [Demo to configure TorchServe ResNet-152 model with batch-supported model using Docker](#demo-to-configure-torchserve-resnet-152-model-with-batch-supported-model-using-docker)

## Introduction

Batch inference is a process of aggregating inference requests and sending this aggregated requests through the ML/DL framework for inference all at once.
TorchServe was designed to natively support batching of incoming inference requests. This functionality enables you to use your host resources optimally,
because most ML/DL frameworks are optimized for batch requests.
This optimal use of host resources in turn reduces the operational expense of hosting an inference service using TorchServe.

In this document we show an example of how to use batch inference in Torchserve when serving models locally or using docker containers. 

## Prerequisites

Before jumping into this document, read the following docs:

1. [What is TorchServe?](../README.md)
1. [What is custom service code?](custom_service.md)

## Batch Inference with TorchServe's default handlers

TorchServe's default handlers support batch inference out of box except for `text_classifier` handler.

## Batch Inference with TorchServe using ResNet-152 model

To support batch inference, TorchServe needs the following:

1. TorchServe model configuration: Configure `batch_size` and `max_batch_delay` by using the  "POST /models" management API or settings in config.properties.
   TorchServe needs to know the maximum batch size that the model can handle and the maximum time that TorchServe should wait to fill each batch request.
2. Model handler code: TorchServe requires the Model handler to handle batch inference requests.

For a full working example of a custom model handler with batch processing, see [Hugging face transformer generalized handler](https://github.com/pytorch/serve/blob/master/examples/Huggingface_Transformers/Transformer_handler_generalized.py)

### TorchServe Model Configuration

Started from Torchserve 0.4.1, there are two methods to configure TorchServe to use the batching feature:
1. provide the batch configuration information through [**POST /models** API](management_api.md).
2. provide the batch configuration information through configuration file, config.properties.

The configuration properties that we are interested in are the following:

1. `batch_size`: This is the maximum batch size that a model is expected to handle.
2. `max_batch_delay`: This is the maximum batch delay time in `ms` TorchServe waits to receive `batch_size` number of requests. If TorchServe doesn't receive `batch_size` number of
requests before this timer time's out, it sends what ever requests that were received to the model `handler`.

Let's look at an example using this configuration through management API:

```bash
# The following command will register a model "resnet-152.mar" and configure TorchServe to use a batch_size of 8 and a max batch delay of 50 milliseconds. 
curl -X POST "localhost:8081/models?url=resnet-152.mar&batch_size=8&max_batch_delay=50"
```
Here is an example of using this configuration through the config.properties:

```text
# The following command will register a model "resnet-152.mar" and configure TorchServe to use a batch_size of 8 and a max batch delay of 50 milli seconds, in the config.properties.

models={\
  "resnet-152": {\
    "1.0": {\
        "defaultVersion": true,\
        "marName": "resnet-152.mar",\
        "minWorkers": 1,\
        "maxWorkers": 1,\
        "batchSize": 8,\
        "maxBatchDelay": 50,\
        "responseTimeout": 120\
    }\
  }\
}

```

These configurations are used both in TorchServe and in the model's custom service code (a.k.a the handler code).
TorchServe associates the batch related configuration with each model.
The frontend then tries to aggregate the batch-size number of requests and send it to the backend.

## Demo to configure TorchServe ResNet-152 model with batch-supported model

In this section lets bring up model server and launch Resnet-152 model, which uses the default `image_classifier` handler for batch inferencing.

### Setup TorchServe and Torch Model Archiver

First things first, follow the main [Readme](../README.md) and install all the required packages including `torchserve`.

### Batch inference of Resnet-152 configured with management API

* Start the model server. In this example, we are starting the model server to run on inference port 8080 and management port 8081.

```text
$ cat config.properties
...
inference_address=http://0.0.0.0:8080
management_address=http://0.0.0.0:8081
...
$ torchserve --start --model-store model_store
```

* Verify that TorchServe is up and running

```text
$ curl localhost:8080/ping
{
  "status": "Healthy"
}
```

* Now let's launch resnet-152 model, which we have built to handle batch inference. Because this is an example, we are going to launch 1 worker which handles a batch size of 3 with a `max_batch_delay` of 10ms.

```text
$ curl -X POST "localhost:8081/models?url=https://torchserve.pytorch.org/mar_files/resnet-152-batch_v2.mar&batch_size=3&max_batch_delay=10&initial_workers=1"
{
  "status": "Processing worker updates..."
}
```

* Verify that the workers were started properly.
```bash
curl http://localhost:8081/models/resnet-152-batch_v2
```

```json
[
  {
    "modelName": "resnet-152-batch_v2",
    "modelVersion": "2.0",
    "modelUrl": "https://torchserve.pytorch.org/mar_files/resnet-152-batch_v2.mar",
    "runtime": "python",
    "minWorkers": 1,
    "maxWorkers": 1,
    "batchSize": 3,
    "maxBatchDelay": 10,
    "loadedAtStartup": false,
    "workers": [
      {
        "id": "9000",
        "startTime": "2021-06-14T23:18:21.793Z",
        "status": "READY",
        "memoryUsage": 1726554112,
        "pid": 19946,
        "gpu": true,
        "gpuUsage": "gpuId::0 utilization.gpu [%]::0 % utilization.memory [%]::0 % memory.used [MiB]::678 MiB"
      }
    ]
  }
]
```

* Now let's test this service.

  * Get an image to test this service

    ```text
    $ curl -LJO https://github.com/pytorch/serve/raw/master/examples/image_classifier/kitten.jpg
    ```

  * Run inference to test the model.

    ```text
      $ curl http://localhost:8080/predictions/resnet-152-batch_v2 -T kitten.jpg
      {
          "tiger_cat": 0.5848360657691956,
          "tabby": 0.3782736361026764,
          "Egyptian_cat": 0.03441936895251274,
          "lynx": 0.0005633446853607893,
          "quilt": 0.0002698268508538604
      }
    ```
### Batch inference of Resnet-152 configured through config.properties

* Here, we first set the `batch_size` and `max_batch_delay`  in the config.properties, make sure the mar file is located in the model-store and the version in the models setting is consistent with version of the mar file created. To read more about configurations please refer to this [document](./configuration.md).

```text
load_models=resnet-152-batch_v2.mar
models={\
  "resnet-152-batch_v2": {\
    "2.0": {\
        "defaultVersion": true,\
        "marName": "resnet-152-batch_v2.mar",\
        "minWorkers": 1,\
        "maxWorkers": 1,\
        "batchSize": 3,\
        "maxBatchDelay": 5000,\
        "responseTimeout": 120\
    }\
  }\
}
```
* Then will start Torchserve by passing the config.properties using `--ts-config` flag 

```bash
torchserve --start --model-store model_store  --ts-config config.properties
```
* Verify that TorchServe is up and running
    
```text
$ curl localhost:8080/ping
{
  "status": "Healthy"
}
```
*  Verify that the workers were started properly.
```bash
curl http://localhost:8081/models/resnet-152-batch_v2
```
```json
[
  {
    "modelName": "resnet-152-batch_v2",
    "modelVersion": "2.0",
    "modelUrl": "resnet-152-batch_v2.mar",
    "runtime": "python",
    "minWorkers": 1,
    "maxWorkers": 1,
    "batchSize": 3,
    "maxBatchDelay": 5000,
    "loadedAtStartup": true,
    "workers": [
      {
        "id": "9000",
        "startTime": "2021-06-14T22:44:36.742Z",
        "status": "READY",
        "memoryUsage": 0,
        "pid": 19116,
        "gpu": true,
        "gpuUsage": "gpuId::0 utilization.gpu [%]::0 % utilization.memory [%]::0 % memory.used [MiB]::678 MiB"
      }
    ]
  }
]
```
* Now let's test this service.

  * Get an image to test this service

    ```text
    $ curl -LJO https://github.com/pytorch/serve/raw/master/examples/image_classifier/kitten.jpg
    ```

  * Run inference to test the model.

    ```text
      $ curl http://localhost:8080/predictions/resnet-152-batch_v2 -T kitten.jpg
      {
          "tiger_cat": 0.5848360657691956,
          "tabby": 0.3782736361026764,
          "Egyptian_cat": 0.03441936895251274,
          "lynx": 0.0005633446853607893,
          "quilt": 0.0002698268508538604
      }
    ```
## Demo to configure TorchServe ResNet-152 model with batch-supported model using Docker

Here, we show how to register a model with batch inference support when serving the model using docker containers. We set the `batch_size` and `max_batch_delay`  in the config.properties similar to the previous section which is being used by [dockered_entrypoint.sh](../docker/dockerd-entrypoint.sh).

### Batch inference of Resnet-152 using docker container

* Set the batch `batch_size` and `max_batch_delay`  in the config.properties as referenced in the [dockered_entrypoint.sh](../docker/dockerd-entrypoint.sh)

```text
inference_address=http://0.0.0.0:8080
management_address=http://0.0.0.0:8081
metrics_address=http://0.0.0.0:8082
number_of_netty_threads=32
job_queue_size=1000
model_store=/home/model-server/model-store
load_models=resnet-152-batch_v2.mar
models={\
  "resnet-152-batch_v2": {\
    "1.0": {\
        "defaultVersion": true,\
        "marName": "resnet-152-batch_v2.mar",\
        "minWorkers": 1,\
        "maxWorkers": 1,\
        "batchSize": 3,\
        "maxBatchDelay": 100,\
        "responseTimeout": 120\
    }\
  }\
}
```
* build the targeted docker image from [here](../docker), here we use the gpu image
```bash
./build_image.sh -g -cv cu102
```

* Start serving the model with the container and pass the config.properties to the container 

```bash
 docker run --rm -it --gpus all -p 8080:8080 -p 8081:8081 --name mar -v /home/ubuntu/serve/model_store:/home/model-server/model-store  -v $ path to config.properties:/home/model-server/config.properties  pytorch/torchserve:latest-gpu
```
* Verify that the workers were started properly.
```bash
curl http://localhost:8081/models/resnet-152-batch_v2
```
```json
[
  {
    "modelName": "resnet-152-batch_v2",
    "modelVersion": "2.0",
    "modelUrl": "resnet-152-batch_v2.mar",
    "runtime": "python",
    "minWorkers": 1,
    "maxWorkers": 1,
    "batchSize": 3,
    "maxBatchDelay": 5000,
    "loadedAtStartup": true,
    "workers": [
      {
        "id": "9000",
        "startTime": "2021-06-14T22:44:36.742Z",
        "status": "READY",
        "memoryUsage": 0,
        "pid": 19116,
        "gpu": true,
        "gpuUsage": "gpuId::0 utilization.gpu [%]::0 % utilization.memory [%]::0 % memory.used [MiB]::678 MiB"
      }
    ]
  }
]
```
* Now let's test this service.

  * Get an image to test this service

    ```text
    $ curl -LJO https://github.com/pytorch/serve/raw/master/examples/image_classifier/kitten.jpg
    ```

  * Run inference to test the model.

    ```text
      $ curl http://localhost:8080/predictions/resnet-152-batch_v2 -T kitten.jpg
      {
          "tiger_cat": 0.5848360657691956,
          "tabby": 0.3782736361026764,
          "Egyptian_cat": 0.03441936895251274,
          "lynx": 0.0005633446853607893,
          "quilt": 0.0002698268508538604
      }
    ```
