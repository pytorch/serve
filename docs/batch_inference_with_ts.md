# Batch Inference with TorchServe

## Contents of this Document

* [Introduction](#introduction)
* [Prerequisites](#prerequisites)
* [Batch Inference with TorchServe's default handlers](#batch-inference-with-torchserves-default-handlers)
* [Batch Inference with TorchServe using ResNet-152 model](#batch-inference-with-torchserve-using-resnet-152-model)   
* [Conclusion](#conclusion)   

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

TorchServe's default handlers do not support batch inference.

## Batch Inference with TorchServe using ResNet-152 model

To support batch inference, TorchServe needs the following:

1. TorchServe model configuration: Configure `batch_size` and `max_batch_delay` by using the  "POST /models" management API.
   TorchServe needs to know the maximum batch size that the model can handle and the maximum time that TorchServe should wait to fill each batch request.
2. Model handler code: TorchServe requires the Model handler to handle batch inference requests.

For a full working example of a custom model handler with batch processing, see [resnet152_handler.py](../examples/image_classifier/resnet_152_batch/resnet152_handler.py)

### TorchServe Model Configuration

To configure TorchServe to use the batching feature, provide the batch configuration information through [**POST /models** API](management_api.md#register-a-model).

The configuration that we are interested in is the following:

1. `batch_size`: This is the maximum batch size that a model is expected to handle.
2. `max_batch_delay`: This is the maximum batch delay time TorchServe waits to receive `batch_size` number of requests. If TorchServe doesn't receive `batch_size` number of
requests before this timer time's out, it sends what ever requests that were received to the model `handler`.

Let's look at an example using this configuration

```bash
# The following command will register a model "resnet-152.mar" and configure TorchServe to use a batch_size of 8 and a max batch delay of 50 milli seconds. 
curl -X POST "localhost:8081/models?url=resnet-152.mar&batch_size=8&max_batch_delay=50"
```

These configurations are used both in TorchServe and in the model's custom service code (a.k.a the handler code).
TorchServe associates the batch related configuration with each model.
The frontend then tries to aggregate the batch-size number of requests and send it to the backend.

## Demo to configure TorchServe with batch-supported model

In this section lets bring up model server and launch Resnet-152 model, which has been built to handle a batch of request.

### Prerequisites

Follow the main [Readme](../README.md) and install all the required packages including `torchserve`.

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

**Note**: This example assumes that the resnet-152.mar file is available in the `model_store`.
For more details on creating resnet-152 mar file and serving it on TorchServe refer [resnet152 image classification example](../examples/image_classifier/resnet_152_batch/README.md)

* Verify that TorchServe is up and running

```text
$ curl localhost:8080/ping
{
  "status": "Healthy"
}
```

* Now let's launch resnet-152 model, which we have built to handle batch inference. Because this is an example, we are going to launch 1 worker which handles a batch size of 8 with a `max_batch_delay` of 10ms.

```text
$ curl -X POST "localhost:8081/models?url=resnet-152.mar&batch_size=8&max_batch_delay=10&initial_workers=1"
{
  "status": "Processing worker updates..."
}
```

* Verify that the workers were started properly.

```text
$ curl localhost:8081/models/resnet-152
{
  "modelName": "resnet-152",
  "modelUrl": "https://s3.amazonaws.com/model-server/model_archive_1.0/examples/resnet-152-batching/resnet-152.mar",
  "runtime": "python",
  "minWorkers": 1,
  "maxWorkers": 1,
  "batchSize": 8,
  "maxBatchDelay": 10,
  "workers": [
    {
      "id": "9008",
      "startTime": "2019-02-19T23:56:33.907Z",
      "status": "READY",
      "gpu": false,
      "memoryUsage": 607715328
    }
  ]
}
```

* Now let's test this service.

  * Get an image to test this service

    ```text
    $ curl -O https://s3.amazonaws.com/model-server/inputs/kitten.jpg
    ```

  * Run inference to test the model.

    ```text
      $ curl localhost/predictions/resnet-152 -T kitten.jpg
      {
        "probability": 0.7148938179016113,
        "class": "n02123045 tabby, tabby cat"
      },
      {
        "probability": 0.22877725958824158,
        "class": "n02123159 tiger cat"
      },
      {
        "probability": 0.04032370448112488,
        "class": "n02124075 Egyptian cat"
      },
      {
        "probability": 0.00837081391364336,
        "class": "n02127052 lynx, catamount"
      },
      {
        "probability": 0.0006728120497427881,
        "class": "n02129604 tiger, Panthera tigris"
      }
    ```

* Now that we have the service up and running, we can run performance tests with the same kitten image as follows. There are multiple tools to measure performance of web-servers. We will use
[apache-bench](https://httpd.apache.org/docs/2.4/programs/ab.html) to run our performance tests. We chose `apache-bench` for our tests because of the ease of installation and ease of running tests.

Before running this test, we need to first install `apache-bench` on our system. Since we were running this on an Ubuntu host, we install `apache-bench` as follows:

```bash
$ sudo apt-get update && sudo apt-get install apache2-utils
```

Now that installation is done, we can run performance benchmark test as follows.

```text
$ ab -k -l -n 10000 -c 1000 -T "image/jpeg" -p kitten.jpg localhost:8080/predictions/resnet-152
```

The above test simulates TorchServe receiving 1000 concurrent requests at once and a total of 10,000 requests. All of these requests are directed to the endpoint "localhost:8080/predictions/resnet-152", which assumes
that resnet-152 is already registered and scaled-up on TorchServe. We had done this registration and scaling up in the above steps.

## Conclusion

The take away from this example is that batching is a very useful feature. In cases where the services receive heavy load of requests or each request has high I/O,
it's advantageous to batch the requests. This allows for maximally utilizing the compute resources, especially GPU resources, which are more expensive. But customers should do their due diligence and perform enough tests to find optimal batch size depending on the number of GPUs available
and number of models loaded per GPU.
You should also analyze your traffic patterns before enabling the batch inference. As shown in the above experiments,
services receiving TPS less than than the batch size would lead to consistent "batch delay" timeouts and cause the response latency per request to spike.
As with any cutting-edge technology, batch inference is definitely a double-edged sword.
