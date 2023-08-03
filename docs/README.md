# TorchServe

TorchServe is a performant, flexible and easy to use tool for serving PyTorch eager mode and torchscripted models.

## Basic Features

* [Serving Quick Start](https://github.com/pytorch/serve/blob/master/README.md#serve-a-model) - Basic server usage tutorial
* [Model Archive Quick Start](https://github.com/pytorch/serve/tree/master/model-archiver#creating-a-model-archive) - Tutorial that shows you how to package a model archive file.
* [Installation](https://github.com/pytorch/serve/blob/master/README.md#install-torchserve) - Installation procedures
* [Serving Models](server.md) - Explains how to use TorchServe
* [REST API](rest_api.md) - Specification on the API endpoint for TorchServe
* [gRPC API](grpc_api.md) - TorchServe supports gRPC APIs for both inference and management calls
* [Packaging Model Archive](https://github.com/pytorch/serve/tree/master/model-archiver#torch-model-archiver-for-torchserve) - Explains how to package model archive file, use `model-archiver`.
* [Inference API](inference_api.md) - How to check for the health of a deployed model and get inferences
* [Management API](management_api.md) - How to manage and scale models
* [Logging](logging.md) - How to configure logging
* [Metrics](metrics.md) - How to configure metrics
* [Prometheus and Grafana metrics](metrics_api.md) - How to configure metrics API with Prometheus formatted metrics in a Grafana dashboard
* [Captum Explanations](https://github.com/pytorch/serve/blob/master/examples/captum/Captum_visualization_for_bert.ipynb) - Built in support for Captum explanations for both text and images
* [Batch inference with TorchServe](batch_inference_with_ts.md) - How to create and serve a model with batch inference in TorchServe
* [Workflows](workflows.md) - How to create workflows to compose Pytorch models and Python functions in sequential and parallel pipelines



## Default Handlers

* [Image Classifier](https://github.com/pytorch/serve/blob/master/ts/torch_handler/image_classifier.py) - This handler takes an image and returns the name of object in that image
* [Text Classifier](https://github.com/pytorch/serve/blob/master/ts/torch_handler/text_classifier.py) - This handler takes a text (string) as input and returns the classification text based on the model vocabulary
* [Object Detector](https://github.com/pytorch/serve/blob/master/ts/torch_handler/object_detector.py) - This handler takes an image and returns list of detected classes and bounding boxes respectively
* [Image Segmenter](https://github.com/pytorch/serve/blob/master/ts/torch_handler/image_segmenter.py)- This handler takes an image and returns output shape as [CL H W], CL - number of classes, H - height and W - width

## Examples

* [HuggingFace Language Model](https://github.com/pytorch/serve/blob/master/examples/Huggingface_Transformers/Transformer_handler_generalized.py) - This handler takes an input sentence and can return sequence classifications, token classifications or Q&A answers
* [Multi Modal Framework](https://github.com/pytorch/serve/blob/master/examples/MMF-activity-recognition/handler.py) - Build and deploy a classifier that combines text, audio and video input data
* [Dual Translation Workflow](https://github.com/pytorch/serve/tree/master/examples/Workflows/nmt_transformers_pipeline) -
* [Model Zoo](model_zoo.md) - List of pre-trained model archives ready to be served for inference with TorchServe.
* [Examples](https://github.com/pytorch/serve/tree/master/examples) - Many examples of how to package and deploy models with TorchServe
     - [TorchServe Internals](../examples/README.md#torchserve-internals)
     - [TorchServe Integrations](../examples/README.md#torchserve-integrations)
     - [TorchServe UseCases](../examples/README.md#usecases)
* [Workflow Examples](https://github.com/pytorch/serve/tree/master/examples/Workflows) - Examples of how to compose models in a workflow with TorchServe

## Advanced Features

* [Advanced configuration](configuration.md) - Describes advanced TorchServe configurations.
* [A/B test models](https://github.com/pytorch/serve/blob/master/docs/use_cases.md#serve-models-for-ab-testing) - A/B test your models for regressions before shipping them to production
* [Custom Service](custom_service.md) - Describes how to develop custom inference services.
* [Encrypted model serving](https://github.com/pytorch/serve/blob/master/docs/management_api.md#encrypted-model-serving) - S3 server side model encryption via KMS
* [Snapshot serialization](https://github.com/pytorch/serve/blob/master/plugins/docs/ddb_endpoint.md) - Serialize model artifacts to AWS Dynamo DB
* [Benchmarking and Profiling](https://github.com/pytorch/serve/tree/master/benchmarks#torchserve-model-server-benchmarking) - Use JMeter or Apache Bench to benchmark your models and TorchServe itself
* [TorchServe on Kubernetes](https://github.com/pytorch/serve/blob/master/kubernetes/README.md#torchserve-on-kubernetes) -  Demonstrates a Torchserve deployment in Kubernetes using Helm Chart supported in both Azure Kubernetes Service and Google Kubernetes service
* [mlflow-torchserve](https://github.com/mlflow/mlflow-torchserve) - Deploy mlflow pipeline models into TorchServe
* [Kubeflow pipelines](https://github.com/kubeflow/pipelines/tree/master/samples/contrib/pytorch-samples) - Kubeflow pipelines and Google Vertex AI Managed pipelines
* [NVIDIA MPS](mps.md) - Use NVIDIA MPS to optimize multi-worker deployment on a single GPU
