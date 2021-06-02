# Batch Inference with TorchServe

## Contents of this Document

* [Introduction](#introduction)
* [Prerequisites](#prerequisites)
* [Batch Inference with TorchServe's default handlers](#batch-inference-with-torchserves-default-handlers)
* [Batch Inference with TorchServe using ResNet-152 model](#batch-inference-with-torchserve-using-resnet-152-model)  

## Introduction

Batch inference is a process of aggregating inference requests and sending this aggregated requests through the ML/DL framework for inference all at once.
TorchServe was designed to natively support batching of incoming inference requests. This functionality enables you to use your host resources optimally,
because most ML/DL frameworks are optimized for batch requests.
This optimal use of host resources in turn reduces the operational expense of hosting an inference service using TorchServe.
In this document we show an example of how this is done and compare the performance of running a batched inference against running single inference.

## Prerequisites

Before jumping into this document, read the following docs:

1. [What is TorchServe?](../README.md)
1. [What is custom service code?](custom_service.md)

## Batch Inference with TorchServe's default handlers

TorchServe's default handlers support batch inference out of box except for `text_classifier` handler.

## Batch Inference with TorchServe using ResNet-152 model

To support batch inference, TorchServe needs the following:

1. TorchServe model configuration: Configure `batch_size` and `max_batch_delay` by using the  "POST /models" management API.
   TorchServe needs to know the maximum batch size that the model can handle and the maximum time that TorchServe should wait to fill each batch request.
2. Model handler code: TorchServe requires the Model handler to handle batch inference requests.

For a full working example of a custom model handler with batch processing, see [Hugging face transformer generalized handler](https://github.com/pytorch/serve/blob/master/examples/Huggingface_Transformers/Transformer_handler_generalized.py)

### TorchServe Model Configuration

To configure TorchServe to use the batching feature, provide the batch configuration information through [**POST /models** API](management_api.md).

The configuration that we are interested in is the following:

1. `batch_size`: This is the maximum batch size that a model is expected to handle.
2. `max_batch_delay`: This is the maximum batch delay time TorchServe waits to receive `batch_size` number of requests. If TorchServe doesn't receive `batch_size` number of
requests before this timer time's out, it sends what ever requests that were received to the model `handler`.

Let's look at an example using this configuration

```bash
# The following command will register a model "resnet-152.mar" and configure TorchServe to use a batch_size of 8 and a max batch delay of 50 milliseconds. 
curl -X POST "localhost:8081/models?url=resnet-152.mar&batch_size=8&max_batch_delay=50"
```

These configurations are used both in TorchServe and in the model's custom service code (a.k.a the handler code).
TorchServe associates the batch related configuration with each model.
The frontend then tries to aggregate the batch-size number of requests and send it to the backend.

## Demo to configure TorchServe with batch-supported model

In this section lets bring up model server and launch Resnet-152 model, which uses the default `image_classifier` handler for batch inferencing.

### Setup TorchServe and Torch Model Archiver

First things first, follow the main [Readme](../README.md) and install all the required packages including `torchserve`.

### Loading Resnet-152 which handles batch inferences

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

* Now let's launch resnet-152 model, which we have built to handle batch inference. Because this is an example, we are going to launch 1 worker which handles a batch size of 8 with a `max_batch_delay` of 10ms.

```text
$ curl -X POST "localhost:8081/models?url=https://torchserve.pytorch.org/mar_files/resnet-152-batch_v2.mar&batch_size=8&max_batch_delay=10&initial_workers=1"
{
  "status": "Processing worker updates..."
}
```

* Verify that the workers were started properly.

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
    "maxBatchDelay": 5000,
    "loadedAtStartup": false,
    "workers": [
      {
        "id": "9000",
        "startTime": "2020-07-28T05:04:05.465Z",
        "status": "READY",
        "gpu": false,
        "memoryUsage": 0
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
