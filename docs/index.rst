TorchServe
==================================

TorchServe is a flexible and easy to use tool for serving PyTorch models.

.. warning ::
     TorchServe is experimental and subject to change.


Basic Features
----------------

* :ref:`Serving Quick Start <install:Serve a model>` - Basic server usage tutorial
* :ref:`Model Archive Quick Start <model-archiver:Creating a Model Archive>` - Tutorial that shows you how to package a model archive file.
* :ref:`Installation <install:Install and Serve>` - Installation procedures
* :ref:`Serving Models <server:Running TorchServe>` - Explains how to use `torchserve`.
   * :ref:`REST API <rest_api:TorchServe REST API>` - Specification on the API endpoint for TorchServe
* :ref:`Packaging Model Archive <model-archiver:Torch Model archiver for TorchServe>` - Explains how to package model archive file, use `model-archiver`.
* :ref:`Logging <logging:Logging in Torchserve>` - How to configure logging
* :ref:`Metrics <metrics:TorchServe Metrics>` - How to configure metrics
* :ref:`Batch inference with TorchServe <batch_inference_with_ts:Batch Inference with TorchServe>` - How to create and serve a model with batch inference in TorchServe

Advanced Features
----------------

* :ref:`Advanced configuration <configuration:Advanced configuration>` - Describes advanced TorchServe configurations.
* :ref:`Custom Service <custom-service:Custom Service>` - Describes how to develop custom inference services.
* `Unit Tests <https://github.com/pytorch/serve/tree/master/ts/tests/README.md>`_ - Housekeeping unit tests for TorchServe.
* `Benchmark <https://github.com/pytorch/serve/blob/master/benchmarks/README.md>`_ - Use JMeter to run TorchServe through the paces and collect benchmark data.

Default Handlers
----------------

* `Image Classifier <https://github.com/pytorch/serve/blob/staging_0_1_1/ts/torch_handler/image_classifier.py>`_ - This handler takes an image and returns the name of object in that image
* `Text Classifier <https://github.com/pytorch/serve/blob/staging_0_1_1/ts/torch_handler/text_classifier.py>`_ - This handler takes a text (string) as input and returns the classification text based on the model vocabulary
* `Object Detector <https://github.com/pytorch/serve/blob/staging_0_1_1/ts/torch_handler/object_detector.py>`_ - This handler takes an image and returns list of detected classes and bounding boxes respectively
* `Image Segmenter <https://github.com/pytorch/serve/blob/staging_0_1_1/ts/torch_handler/image_segmenter.py>`_ - This handler takes an image and returns output shape as [CL H W], CL - number of classes, H - height and W - width
