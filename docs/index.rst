TorchServe
==================================

TorchServe is a flexible and easy to use tool for serving PyTorch models.

.. warning ::
     TorchServe is experimental and subject to change.


Basic Features
----------------

* `Serving Quick Start <https://pytorch.org/serve/install.html#serve-a-model>`_ - Basic server usage tutorial
* `Model Archive Quick Start <https://pytorch.org/serve/model-archiver.html#creating-a-model-archive>`_ - Tutorial that shows you how to package a model archive file.
* `Installation <https://pytorch.org/serve/install.html##install-torchserve>`_ - Installation procedures
* `Serving Models <https://pytorch.org/serve/server.md>`_ - Explains how to use `torchserve`.
   * `REST API <https://pytorch.org/serve/rest_api.md>`_ - Specification on the API endpoint for TorchServe
* `Packaging Model Archive <https://pytorch.org/serve/model-archiver.html/>`_ - Explains how to package model archive file, use `model-archiver`.
* `Logging <https://pytorch.org/serve/logging.md>`_ - How to configure logging
* `Metrics <https://pytorch.org/serve/metrics.md>`_ - How to configure metrics
* `Batch inference with TorchServe <batch_inference_with_ts.md>`_ - How to create and serve a model with batch inference in TorchServe

## Advanced Features

* `Advanced settings <https://pytorch.org/serve/configuration.md>`_ - Describes advanced TorchServe configurations.
* `Custom Model Service <https://pytorch.org/serve/custom_service.md>`_ - Describes how to develop custom inference services.
* `Unit Tests <https://github.com/pytorch/serve/tree/master/ts/tests/README.md>`_ - Housekeeping unit tests for TorchServe.
* `Benchmark <https://github.com/pytorch/serve/blob/master/benchmarks/README.md>`_ - Use JMeter to run TorchServe through the paces and collect benchmark data.
