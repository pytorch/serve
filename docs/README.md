# Model Server for Apache MXNet Documentation

## Basic Features
* [Serving Quick Start](../README.md#serve-a-model) - Basic server usage tutorial
* [Model Archive Quick Start](../model-archiver#creating-a-model-archive) - Tutorial that shows you how to package a model archive file.
* [Installation](install.md) - Installation procedures and troubleshooting
* [Serving Models](server.md) - Explains how to use `mxnet-model-server`.
  * [REST API](rest_api.md) - Specification on the API endpoint for MMS
  * [Model Zoo](model_zoo.md) - A collection of MMS model archive (.mar) files that you can use with MMS.
* [Packaging Model Archive](../model-archiver/README.md) - Explains how to package model archive file, use `model-archiver`.
* [Docker](../docker/README.md) - How to use MMS with Docker and cloud services
* [Logging](logging.md) - How to configure logging
* [Metrics](metrics.md) - How to configure metrics

## Advanced Features
* [Advanced settings](configuration.md) - Describes advanced MMS configurations.
* [Custom Model Service](custom_service.md) - Describes how to develop custom inference services.
* [Unit Tests](../mms/tests/README.md) - Housekeeping unit tests for MMS.
* [Benchmark](../benchmarks/README.md) - Use JMeter to run MMS through the paces and collect benchmark data.
* [Model Serving with Amazon Elastic Inference](elastic_inference.md) - Run Model server on Elastic Inference enabled EC2 instances. 

## Example Projects
* [MMS on Fargate, Serverless Inference](mms_on_fargate.md) - The project which illustrates the step-by-step process to launch MMS as a managed inference production service, on ECS Fargate.
* [MXNet Vision Service](../examples/mxnet_vision/README.md) - An example MMS project for a MXNet Image Classification model. The project takes JPEG image as input for inference.
* [LSTM](../examples/lstm_ptb/README.md) - An example MMS project for a recurrent neural network (RNN) using long short-term memory (LSTM). The project takes JSON inputs for inference against a model trained with a specific vocabulary.
* [Object Detection](../examples/ssd/README.md) - An example MMS project that uses a pretrained Single Shot Multi Object Detection (SSD) model that takes image inputs and infers the types and locations of several classes of objects.
