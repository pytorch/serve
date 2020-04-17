# Model Server for PyTorch Documentation

## Basic Features

* [Serving Quick Start](../README.md#serve-a-model) - Basic server usage tutorial
* [Model Archive Quick Start](../model-archiver#creating-a-model-archive) - Tutorial that shows you how to package a model archive file.
* [Installation](../README.md##install-torchserve) - Installation procedures
* [Serving Models](server.md) - Explains how to use `torchserve`.
  * [REST API](rest_api.md) - Specification on the API endpoint for TorchServe
* [Packaging Model Archive](../model-archiver/README.md) - Explains how to package model archive file, use `model-archiver`.
* [Logging](logging.md) - How to configure logging
* [Metrics](metrics.md) - How to configure metrics
* [Batch inference with TorchServe](batch_inference_with_ts.md) - How to create and serve a model with batch inference in TorchServe

## Advanced Features

* [Advanced settings](configuration.md) - Describes advanced TorchServe configurations.
* [Custom Model Service](custom_service.md) - Describes how to develop custom inference services.
* [Unit Tests](../ts/tests/README.md) - Housekeeping unit tests for TorchServe.
* [Benchmark](../benchmarks/README.md) - Use JMeter to run TorchServe through the paces and collect benchmark data.
