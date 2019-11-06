# Running the Model Server

## Contents of this Document
* [Overview](#overview)
* [Technical Details](#technical-details)
* [Model Files](#model-files)
* [Command Line Interface](#command-line-interface)
* [Advanced Features](#advanced-features)

## Overview

MXNet Model Server can be used for many types of inference in production settings. It provides an easy-to-use command line interface and utilizes  [REST based APIs](rest_api.md) handle state prediction requests. Support for models from a wide range of deep learning frameworks is achieved through its [ONNX model](https://onnx.ai) export feature.

For example, you want to make an app that lets your users snap a picture, and it will tell them what objects were detected in the scene and predictions on what the objects might be. You can use MMS to serve a prediction endpoint for a object detection and identification model that intakes images, then returns predictions. You can also modify MMS behavior with custom services and run multiple models. There are examples of custom services in the [examples](../examples) folder.

## Technical Details

Now that you have a high level view of MMS, let's get a little into the weeds. MMS takes a deep learning model and it wraps it in a set of REST APIs. Currently it comes with a built-in web server that you run from command line. This command line call takes in the single or multiple models you want to serve, along with additional optional parameters controlling the port, host, and logging. MMS supports running custom services to handle the specific inference handling logic. These are covered in more detail in the [custom service](custom_service.md) documentation.

To try out MMS serving now, you can load the SqueezeNet model, which is under 5 MB, with this example:

```bash
mxnet-model-server --start --models squeezenet=https://s3.amazonaws.com/model-server/model_archive_1.0/squeezenet_v1.1.mar
```

With the command above executed, you have MMS running on your host, listening for inference requests.

To test it out, you will need to open a new terminal window next to the one running MMS. Then we will use `curl` to download one of these [cute pictures of a kitten](https://www.google.com/search?q=cute+kitten&tbm=isch&hl=en&cr=&safe=images) and curl's `-o` flag will name it `kitten.jpg` for us. Then we will `curl` a `POST` to the MMS predictions endpoint with the kitten's image. In the example below, both of these steps are provided.

```bash
curl -o kitten.jpg \
  https://upload.wikimedia.org/wikipedia/commons/8/8f/Cute-kittens-12929201-1600-1200.jpg
curl -X POST http://127.0.0.1:8080/predictions/squeezenet -T kitten.jpg
```

![kitten](https://upload.wikimedia.org/wikipedia/commons/8/8f/Cute-kittens-12929201-1600-1200.jpg)

The predict endpoint will return a prediction response in JSON. Each of the probabilities are percentages, and the classes are coming from a `synset` file included inside the model archive which holds the thousand ImageNet classes this model is matching against. It will look something like the following result, where the 0.94 result is a 94% probable match with an Egyptian cat:


```json
[
  {
    "probability": 0.8582232594490051,
    "class": "n02124075 Egyptian cat"
  },
  {
    "probability": 0.09159987419843674,
    "class": "n02123045 tabby, tabby cat"
  },
  {
    "probability": 0.0374876894056797,
    "class": "n02123159 tiger cat"
  },
  {
    "probability": 0.006165083032101393,
    "class": "n02128385 leopard, Panthera pardus"
  },
  {
    "probability": 0.0031716004014015198,
    "class": "n02127052 lynx, catamount"
  }
]
```
You will see this result in the response to your `curl` call to the predict endpoint, in the terminal window running MMS and log files.

After this deep dive, you might also be interested in:
* [Logging](logging.md): logging options that are available

* [Metrics](metrics.md): details on metrics collection 

* [REST API Description](rest_api.md): more detail about the server's endpoints

* [Model Zoo](model_zoo.md): try serving different models

* [Custom Services](custom_service.md): learn about serving different kinds of model and inference types


## Model Files

The rest of this topic focus on serving of model files without much discussion on the model files themselves, where they come from, and how they're made. Long story short: it's a zip archive with the parameters, weights, and metadata that define a model that has been trained already. If you want to know more about the model files, take a look at the [model-archiver documentation](../model-archiver/README.md).

## Command Line Interface

```bash
$ mxnet-model-server --help
usage: mxnet-model-server [-h] [--start]
                          [--stop]
                          [--mms-config MMS_CONFIG]
                          [--model-store MODEL_STORE]
                          [--models MODEL_PATH1 MODEL_NAME=MODEL_PATH2... [MODEL_PATH1 MODEL_NAME=MODEL_PATH2... ...]]
                          [--log-config LOG_CONFIG]

MXNet Model Server

optional arguments:
  -h, --help            show this help message and exit
  --start               Start the model-server
  --stop                Stop the model-server
  --mms-config MMS_CONFIG
                        Configuration file for model server
  --model-store MODEL_STORE
                        Model store location where models can be loaded
  --models MODEL_PATH1 MODEL_NAME=MODEL_PATH2... [MODEL_PATH1 MODEL_NAME=MODEL_PATH2... ...]
                        Models to be loaded using [model_name=]model_location
                        format. Location can be a HTTP URL, a model archive
                        file or directory contains model archive files in
                        MODEL_STORE.
  --log-config LOG_CONFIG
                        Log4j configuration file for model server
```

#### Arguments:
Example where no models are loaded at start time:

```bash
mxnet-model-server
```

There are no default required arguments to start the server

1. **models**: required, <model_name>=<model_path> pairs.

    a) Model path can be a local file path or URI (s3 link, or http link).
        local file path: path/to/local/model/file or file://root/path/to/model/file
        s3 link: s3://S3_endpoint[:port]/...
        http link: http://hostname/path/to/resource

    b) The model file has .mar extension, it is actually a zip file with a .mar extension packing trained models and model signature files. 

    c) Multiple models loading are also supported by specifying multiple name path pairs.
1. **model-store**: optional, A location where models are stored by default, all models in this location are loaded, the model name is same as archive or folder name.
1. **mms-config**: optional, provide a [configuration](configuration.md) file in config.properties format.
1. **log-config**: optional, This parameter will override default log4j.properties, present within the server.
1. **start**: optional, A more descriptive way to start the server.
1. **stop**: optional, Stop the server if it is already running.

## Advanced Features

### Custom Services

This topic is covered in much more detail on the [custom service documentation page](custom_service.md), but let's talk about how you start up your MMS server using a custom service and why you might want one.
Let's say you have a model named `super-fancy-net.mar` in `/models` folder, which can detect a lot of things, but you want an API endpoint that detects only hotdogs. You would use a name that makes sense for it, such as the "not-hot-dog" API. In this case we might invoke MMS like this:

```bash
mxnet-model-server --start  --model-store /models --models not-hot-dog=super-fancy-net.mar
```

This would serve a prediction endpoint at `predictions/not-hot-dog/` and run your custom service code in the archive, the manifest in archive would point to the entry point.

### Serving Multiple Models with MMS

Example multiple model usage:

```bash
mxnet-model-server --start --model-store /models --models name=model_location name2=model_location2
```

Here's an example for running the resnet-18 and the vgg16 models using local model files.

```bash
mxnet-model-server --start --model-store /models --models resnet-18=resnet-18.mar squeezenet=squeezenet_v1.1.mar
```

If you don't have the model files locally, then you can call MMS using URLs to the model files.

```bash
mxnet-model-server --models resnet=https://s3.amazonaws.com/model-server/model_archive_1.0/resnet-18.mar squeezenet=https://s3.amazonaws.com/model-server/model_archive_1.0/squeezenet_v1.1.mar
```

This will setup a local host serving resnet-18 model and squeezenet model on the same port, using the default 8080. Check http://127.0.0.1:8081/models to see that each model has an endpoint for prediction. In this case you would see `predictions/resnet` and `predictions/squeezenet`


### Logging and Metrics

For details on logging see the [logging documentation](logging.md). For details on metrics see the [metrics documentation](metrics.md).


