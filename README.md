# TorchServe

TorchServe is a flexible and easy to use tool for serving PyTorch models in production.

**For full documentation, see [Model Server for PyTorch Documentation](docs/README.md).**

<!-- ## Contents of this Document

* [Install TorchServe](#install-torchserve-and-torch-model-archiver)
* [Install TorchServe on Windows](docs/torchserve_on_win_native.md)
* [Install TorchServe on Windows Subsystem for Linux](docs/torchserve_on_wsl.md)
* [Serve a Model](#serve-a-model)
* [Serve a Workflow](docs/workflows.md)
* [Quick start with docker](#quick-start-with-docker)
* [Highlighted Examples](#highlighted-examples)
* [Featured Community Projects](#featured-community-projects)
* [Contributing](#contributing) -->

## Why TorchServe

Add lots of links clearly

* Multi model management 
* Default way to serve PyTorch models in Kubeflow, Sagemaker, Vertex AI
* Large coverage for model optimization runtimes like Torchscript, ORT, IPEX, TensorRT, FasterTransformer
* Easy to use and expressive model handler based design


### Install TorchServe for development

If you plan to develop with TorchServe and change some source code, you must install it from source code.

Ensure that you have `python3` installed, and the user has access to the site-packages or `~/.local/bin` is added to the `PATH` environment variable.

Run the following script from the top of the source directory.

NOTE: This script uninstalls existing `torchserve`, `torch-model-archiver` and `torch-workflow-archiver` installations

#### For Debian Based Systems/ MacOS

```
python ./ts_scripts/install_dependencies.py --environment=dev
python ./ts_scripts/install_from_src.py
```

Use `--cuda` flag with `install_dependencies.py` for installing cuda version specific dependencies. Possible values are `cu111`, `cu102`, `cu101`, `cu92`

#### For Windows

Refer to the documentation [here](docs/torchserve_on_win_native.md).

For information about the model archiver, see [detailed documentation](model-archiver/README.md).




## Quick Start with Docker
Refer to [torchserve docker](docker/README.md) for details.

## Highlighted Examples
* [HuggingFace Transformers](examples/Huggingface_Transformers)
* [MultiModal models with MMF](https://github.com/pytorch/serve/tree/master/examples/MMF-activity-recognition) combining text, audio and video
* Complex workflows, models chained together in a dependency graph
  - [Dog Breed Classification](examples/Workflows/dog_breed_classification)
  - [Neural Machine Translation](examples/Workflows/nmt_tranformers_pipeline)

Feel free to skim the full list of [available examples](examples/README.md)

## Featured Community Projects
* [SynthDet by Unity Technologies](https://github.com/Unity-Technologies/SynthDet)
* [Torchserve dashboard by cceyda](https://github.com/cceyda/torchserve-dashboard)
* Change this section to include integrations with Vertex AI, Sagemaker, Kubeflow etc..

## Learn More

* [Full documentation on TorchServe](docs/README.md)
* [Model Management API](docs/management_api.md)
* [Inference API](docs/inference_api.md)
* [Metrics API](docs/metrics.md)
* [Package models for use with TorchServe](model-archiver/README.md)
* [Deploying TorchServe with Kubernetes](kubernetes/README.md)
* [TorchServe Workflows](examples/Workflows/README.md)
* [TorchServe model zoo for pre-trained and pre-packaged models-archives](docs/model_zoo.md)

## Contributing

We welcome all contributions!

To learn more about how to contribute, see the contributor guide [here](https://github.com/pytorch/serve/blob/master/CONTRIBUTING.md).

To file a bug or request a feature, please file a GitHub issue. For filing pull requests, please use the template [here](https://github.com/pytorch/serve/blob/master/pull_request_template.md). Cheers!

## Disclaimer 
This repository is jointly operated and maintained by Amazon, Meta and a number of individual contributors listed in the [CONTRIBUTORS](https://github.com/pytorch/serve/graphs/contributors) file. For questions directed at Meta, please send an email to opensource@fb.com. For questions directed at Amazon, please send an email to torchserve@amazon.com. For all other questions, please open up an issue in this repository [here](https://github.com/pytorch/serve/issues).

*TorchServe acknowledges the [Multi Model Server (MMS)](https://github.com/awslabs/multi-model-server) project from which it was derived*
