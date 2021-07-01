# TorchServe

TorchServe is a flexible and easy to use tool for serving PyTorch models.

## Basic Features

* [Serving Quick Start](https://github.com/pytorch/serve/blob/master/README.md#serve-a-model) - Basic server usage tutorial
* [Model Archive Quick Start](https://github.com/pytorch/serve/tree/master/model-archiver#creating-a-model-archive) - Tutorial that shows you how to package a model archive file.
* [Installation](https://github.com/pytorch/serve/blob/master/README.md#install-torchserve) - Installation procedures
* [Serving Models](server.md) - Explains how to use torchserve
   * [REST API](rest_api.md) - Specification on the API endpoint for TorchServe
* [Packaging Model Archive](https://github.com/pytorch/serve/tree/master/model-archiver#torch-model-archiver-for-torchserve) - Explains how to package model archive file, use `model-archiver`.
* [Inference API](inference_api.md) - How to check for the health of a deployed model and get inferences
* [Management API](management_api.md) - How to manage and scale models
* [Logging](logging.md) - How to configure logging
* [Metrics](metrics.md) - How to configure metrics
   * [Metrics API](metrics_api.md) - How to configure metrics API
* [Batch inference with TorchServe](batch_inference_with_ts.md) - How to create and serve a model with batch inference in TorchServe
* [Model Zoo](model_zoo.md) - List of pre-trained model archives ready to be served for inference with TorchServe.
* [Examples](https://github.com/pytorch/serve/tree/master/examples) - Many examples of how to package and deploy models and workflows with TorchServe

## Advanced Features

* [Advanced configuration](configuration.md) - Describes advanced TorchServe configurations.
* [Custom Service](custom_service.md) - Describes how to develop custom inference services.
* [Unit Tests](https://github.com/pytorch/serve/tree/master/ts/tests#testing-torchserve) - Housekeeping unit tests for TorchServe.
* [Benchmark](https://github.com/pytorch/serve/tree/master/benchmarks#torchserve-model-server-benchmarking) - Use JMeter to run TorchServe through the paces and collect benchmark data.
* [TorchServe on Kubernetes](https://github.com/pytorch/serve/blob/master/kubernetes/README.md#torchserve-on-kubernetes) -  Demonstrates a Torchserve deployment in Kubernetes using Helm Chart.

## Default Handlers

* [Image Classifier](https://github.com/pytorch/serve/blob/master/ts/torch_handler/image_classifier.py) - This handler takes an image and returns the name of object in that image
* [Text Classifier](https://github.com/pytorch/serve/blob/master/ts/torch_handler/text_classifier.py) - This handler takes a text (string) as input and returns the classification text based on the model vocabulary
* [Object Detector](https://github.com/pytorch/serve/blob/master/ts/torch_handler/object_detector.py) - This handler takes an image and returns list of detected classes and bounding boxes respectively
* [Image Segmenter](https://github.com/pytorch/serve/blob/master/ts/torch_handler/image_segmenter.py)- This handler takes an image and returns output shape as [CL H W], CL - number of classes, H - height and W - width
