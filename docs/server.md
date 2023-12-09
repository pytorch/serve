# Running TorchServe

## Contents of this Document

* [Overview](#overview)
* [Technical Details](#technical-details)
* [Model Files](#model-files)
* [Command Line Interface](#command-line-interface)
* [Advanced Features](#advanced-features)

## Overview

TorchServe can be used for many types of inference in production settings. It provides an easy-to-use command line interface and utilizes  [REST based APIs](rest_api.md) handle state prediction requests.

For example, you want to make an app that lets your users snap a picture, and it will tell them what objects were detected in the scene and predictions on what the objects might be. You can use TorchServe to serve a prediction endpoint for a object detection and identification model that intakes images, then returns predictions. You can also modify TorchServe behavior with custom services and run multiple models. There are examples of custom services in the [examples](https://github.com/pytorch/serve/tree/master/examples) folder.

## Technical Details

Now that you have a high level view of TorchServe, let's get a little into the weeds. TorchServe takes a Pytorch deep learning model and it wraps it in a set of REST APIs. Currently it comes with a built-in web server that you run from command line. This command line call takes in the single or multiple models you want to serve, along with additional optional parameters controlling the port, host, and logging. TorchServe supports running custom services to handle the specific inference handling logic. These are covered in more detail in the [custom service](custom_service.md) documentation.

To try out TorchServe serving now, you can load the custom MNIST model, with this example:

* [Digit recognition with MNIST](https://github.com/pytorch/serve/tree/master/examples/image_classifier/mnist)

After this deep dive, you might also be interested in:
* [Logging](logging.md): logging options that are available

* [Metrics](metrics.md): details on metrics collection 

* [REST API Description](rest_api.md): more detail about the server's endpoints

* [Custom Services](custom_service.md): learn about serving different kinds of model and inference types


## Model Files

The rest of this topic focuses on serving model files without much discussion on the model files themselves, where they come from, and how they're made. Long story short: it's a zip archive with the parameters, weights, and metadata that define a model that has been trained already. If you want to know more about the model files, take a look at the [model-archiver documentation](https://github.com/pytorch/serve/tree/master/model-archiver).

## Command Line Interface

```bash
$ torchserve --help
usage: torchserve [-h] [-v | --version]
                          [--start]
                          [--stop]
                          [--ts-config TS_CONFIG]
                          [--model-store MODEL_STORE]
                          [--workflow-store WORKFLOW_STORE]
                          [--models MODEL_PATH1 MODEL_NAME=MODEL_PATH2... [MODEL_PATH1 MODEL_NAME=MODEL_PATH2... ...]]
                          [--log-config LOG_CONFIG]

torchserve

optional arguments:
  -h, --help            show this help message and exit
  -v, --version         Return TorchServe Version
  --start               Start the model-server
  --stop                Stop the model-server
  --ts-config TS_CONFIG
                        Configuration file for TorchServe
  --model-store         MODEL_STORE
                        Model store location where models can be loaded.
                        It is required if "model_store" is not defined in config.properties.
  --models MODEL_PATH1 MODEL_NAME=MODEL_PATH2... [MODEL_PATH1 MODEL_NAME=MODEL_PATH2... ...]
                        Models to be loaded using [model_name=]model_location
                        format. Location can be a HTTP URL, a model archive
                        file or directory contains model archive files in
                        MODEL_STORE.
  --log-config LOG_CONFIG
                        Log4j configuration file for TorchServe
  --ncs, --no-config-snapshots         
                        Disable snapshot feature
  --workflow-store WORKFLOW_STORE
                        Workflow store location where workflow can be loaded. Defaults to model-store
```

#### Arguments:

Example where no models are loaded at start time:

```bash
torchserve --model-store /models
```

There are no default required arguments to start the server

1. **models**: optional, <model_name>=<model_path> pairs.

    a) Model path can be a local mar file name or a remote http link to a mar file
    b) to load all the models in model store set model value to "all"


    ```bash
    torchserve --model-store /models --start --models all
    ```

    c) The model file has .mar extension, it is actually a zip file with a .mar extension packing trained models and model signature files.

    d) Multiple models loading are also supported by specifying multiple name path pairs.

    e) For details on different ways to load models while starting TorchServe, refer [Serving Multiple Models with TorchServe](#serving-multiple-models-with-torchserve)

1. **model-store**: mandatory, A location where default or local models are stored. The models available in model store can be registered in TorchServe via [register api call](management_api.md#register-a-model) or via models parameter while starting TorchServe.
1. **workflow-store**: mandatory, A location where default or local workflows are stored. The workflows available in workflow store can be registered in TorchServe via [register api call](workflow_management_api.md#register-a-workflow).
1. **ts-config**: optional, provide a [configuration](configuration.md) file in config.properties format.
1. **log-config**: optional, This parameter will override default log4j2.xml, present within the server.
1. **start**: optional, A more descriptive way to start the server.
1. **stop**: optional, Stop the server if it is already running.

## Advanced Features

### Custom Services

This topic is covered in much more detail on the [custom service documentation page](custom_service.md), but let's talk about how you start up your TorchServe server using a custom service and why you might want one.
Let's say you have a model named `super-fancy-net.mar` in `/models` folder, which can detect a lot of things, but you want an API endpoint that detects only hotdogs. You would use a name that makes sense for it, such as the "not-hot-dog" API. In this case we might invoke TorchServe like this:

```bash
torchserve --start --model-store /models --models not-hot-dog=super-fancy-net.mar
```

This would serve a prediction endpoint at `predictions/not-hot-dog/` and run your custom service code in the archive, the manifest in archive would point to the entry point.

### Serving Multiple Models with TorchServe

Example loading all models available in `model_store` while starting TorchServe:

```bash
torchserve --start --model-store /models --models all
```

Example multiple model usage:

```bash
torchserve --start --model-store /models --models name=model_location name2=model_location2
```

Here's an example for running the resnet-18 and the vgg16 models using local model files.

```bash
torchserve --start --model-store /models --models resnet-18=resnet-18.mar squeezenet=squeezenet_v1.1.mar
```

### Logging and Metrics

For details on logging see the [logging documentation](logging.md). For details on metrics see the [metrics documentation](metrics.md).
