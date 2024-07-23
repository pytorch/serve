# Torch Model archiver for TorchServe

## Contents of this Document
* [Overview](#overview)
* [Installation](#installation)
* [Torch Model Archiver CLI](#torch-model-archiver-command-line-interface)
* [Artifact Details](#artifact-details)
    * [MAR-INFO](#mar-inf)
    * [Model name](#model-name)
    * [Model File](#model-file)
    * [Serialized File](#serialized-file)
    * [handler](#handler)
* [Quick Start: Creating a Model Archive](#creating-a-model-archive)
* [Model specific custom python requirements](#model-specific-custom-python-requirements)

## Overview

A key feature of TorchServe is the ability to package all model artifacts into a single model archive file. It is a separate command line interface (CLI), `torch-model-archiver`, that can take model checkpoints or model definition file with state_dict, and package them into a `.mar` file. This file can then be redistributed and served by anyone using TorchServe. It takes in the following model artifacts: a model checkpoint file in case of torchscript or a model definition file and a state_dict file in case of eager mode, and other optional assets that may be required to serve the model. The CLI creates a `.mar` file that TorchServe's server CLI uses to serve the models.

**Important**: Make sure you try the [Quick Start: Creating a Model Archive](#creating-a-model-archive) tutorial for a short example of using `torch-model-archiver`.

The following information is required to create a standalone model archive:
1. [Model name](#model-name)
2. [Model file](#model-file)
2. [Serialized file](#serialized-file)

## Installation

Install torch-model-archiver as follows:

```bash
pip install torch-model-archiver
```

## Installation from source

Install torch-model-archiver as follows:

```bash
git clone https://github.com/pytorch/serve.git
cd serve/model-archiver
pip install .
```

## Torch Model Archiver Command Line Interface

Now let's cover the details on using the CLI tool: `model-archiver`.

Here is an example usage with the densenet161 model archive following the example in the [examples README](../examples/README.md):

```bash
torch-model-archiver --model-name densenet161 \
    --version 1.0 \
    --model-file examples/image_classifier/densenet_161/model.py \
    --serialized-file densenet161-8d451a50.pth \
    --extra-files examples/image_classifier/index_to_name.json \
    --handler image_classifier
```

### Arguments

```
$ torch-model-archiver -h
usage: torch-model-archiver [-h] --model-name MODEL_NAME  --version MODEL_VERSION_NUMBER
                      --model-file MODEL_FILE_PATH --serialized-file MODEL_SERIALIZED_PATH
                      --handler HANDLER [--runtime {python,python3}]
                      [--export-path EXPORT_PATH] [-f] [--requirements-file] [--config-file]

Model Archiver Tool

optional arguments:
  -h, --help            show this help message and exit
  --model-name MODEL_NAME
                        Exported model name. Exported file will be named as
                        model-name.mar and saved in current working directory
                        if no --export-path is specified, else it will be
                        saved under the export path
  --serialized-file SERIALIZED_FILE
                        Path to .pt or .pth file containing state_dict in
                        case of eager mode or an executable ScriptModule
                        in case of TorchScript.
  --model-file MODEL_FILE
                        Path to python file containing model architecture.
                        This parameter is mandatory for eager mode models.
                        The model architecture file must contain only one
                        class definition extended from torch.nn.Module.
  --handler HANDLER     TorchServe's default handler name  or handler python
                        file path to handle custom TorchServe inference logic.
  --extra-files EXTRA_FILES
                        Comma separated path to extra dependency files.
  --runtime {python,python3}
                        The runtime specifies which language to run your
                        inference code on. The default runtime is
                        RuntimeType.PYTHON. At the present moment we support
                        the following runtimes python, python3
  --export-path EXPORT_PATH
                        Path where the exported .mar file will be saved. This
                        is an optional parameter. If --export-path is not
                        specified, the file will be saved in the current
                        working directory.
  --archive-format {tgz, no-archive, zip-store, default}
                        The format in which the model artifacts are archived.
                        "tgz": This creates the model-archive in <model-name>.tar.gz format.
                        If platform hosting requires model-artifacts to be in ".tar.gz"
                        use this option.
                        "no-archive": This option creates an non-archived version of model artifacts
                        at "export-path/{model-name}" location. As a result of this choice,
                        MANIFEST file will be created at "export-path/{model-name}" location
                        without archiving these model files
                        "zip-store": This creates the model-archive in <model-name>.mar format
                        but will skip deflating the files to speed up creation. Mainly used
                        for testing purposes
                        "default": This creates the model-archive in <model-name>.mar format.
                        This is the default archiving format. Models archived in this format
                        will be readily hostable on TorchServe.
  -f, --force           When the -f or --force flag is specified, an existing
                        .mar file with same name as that provided in --model-
                        name in the path specified by --export-path will
                        overwritten
  -v, --version         Model's version.
  -r, --requirements-file
                        Path to requirements.txt file containing a list of model specific python
                        packages to be installed by TorchServe for seamless model serving.
  -c, --config-file         Path to a model config yaml file.
```

## Artifact Details

### MAR-INF
**MAR-INF** is a reserved folder name that will be used inside `.mar` file. This folder contains the model archive metadata files. Users should avoid using **MAR-INF** in their model path.

### Runtime

### Model name

A valid model name must begin with a letter of the alphabet and can only contains letters, digits, underscores `_`, dashes `-` and periods `.`.

**Note**: The model name can be overridden when you register the model with [Register Model API](../docs/management_api.md#register-a-model).

### Model file

A model file should contain the model architecture. This file is mandatory in case of eager mode models.

This file should contain a single class that inherits from
[torch.nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html).

### Serialized file

A serialized file (.pt or .pth) should be a checkpoint in case of torchscript and state_dict in case of eager mode.

### Handler

Handler can be TorchServe's inbuilt handler name or path to a py file to handle custom TorchServe inference logic. TorchServe supports the following handlers out of box:
1. `image_classifier`
2. `object_detector`
3. `text_classifier`
4. `image_segmenter`

For a more comprehensive list of built in handlers, make sure to checkout the [examples](../docs/default_handlers.md)

In case of custom handler, if you plan to provide just `module_name` or `module_name:entry_point_function_name` then make sure that it is prefixed with absolute or relative path of python file.
e.g. if your custom handler custom_image_classifier.py is in /home/serve/examples then
`--handler /home/serve/examples/custom_image_classifier` or if it has my_entry_point module level function then `--handler /home/serve/examples/custom_image_classifier:my_entry_point_func`

For more details refer [default handler documentation](../docs/default_handlers.md) or [custom handler documentation](../docs/custom_service.md)

### Config file

A model config yaml file. For example:

```
# TS frontend parameters
# See all supported parameters: https://github.com/pytorch/serve/blob/master/frontend/archive/src/main/java/org/pytorch/serve/archive/model/ModelConfig.java#L14
minWorkers: 1 # default: #CPU or #GPU
maxWorkers: 1 # default: #CPU or #GPU
batchSize: 1 # default: 1
maxBatchDelay: 100 # default: 100 msec
responseTimeout: 120 # default: 120 sec
deviceType: cpu # cpu, gpu, neuron
deviceIds: [0,1,2,3] # gpu device ids allocated to this model.
parallelType: pp # pp: pipeline parallel; pptp: tensor+pipeline parallel; custom: user defined parallelism. Default: empty
# parallelLevel: 1 # number of GPUs assigned to a the worker process if parallelType is custom (do NOT set if torchrun is used, see below)
useVenv: Create python virtual environment when using python backend to install model dependencies
         (if enabled globally using install_py_dep_per_model=true) and run workers for model loading
         and inference. Note that, although creation of virtual environment adds a latency overhead
         (approx. 2 to 3 seconds) during model load and disk space overhead (approx. 25M), overall
         it can speed up load time and reduce disk utilization for models with custom dependencies
         since it enables reusing custom packages(specified in requirements.txt) and their
         supported dependencies that are already available in the base python environment.

# See torchrun parameters: https://pytorch.org/docs/stable/elastic/run.html
torchrun:
  nproc-per-node: 2

# TS backend parameters
pippy:
  rpc_timeout: 1800
  pp_group_size: 4 # pipeline parallel size, tp_group_size = world size / pp_group_size
```

## Creating a Model Archive

**1. Download the torch model archiver source**
```bash
git clone https://github.com/pytorch/serve.git
```

**2. Package your model**

With the model artifacts available locally, you can use the `torch-model-archiver` CLI to generate a `.mar` file that can be used to serve an inference API with TorchServe.

In this next step we'll run `torch-model-archiver` and tell it our model's name is `densenet_161` and its version is `1.0` with the `model-name` and `version` parameter respectively and that it will use TorchServe's default `image_classifier` handler with the `handler` argument . Then we're giving it the `model-file` and `serialized-file` to the model's assets.

For torchscript:
```bash
torch-model-archiver --model-name densenet_161 --version 1.0 --serialized-file model.pt --handler image_classifier
```

For eagermode:
```bash
torch-model-archiver --model-name densenet_161 --version 1.0 --model-file model.py --serialized-file model.pt --handler image_classifier
```

This will package all the model artifacts files and output `densenet_161.mar` in the current working directory. This `.mar` file is all you need to run TorchServe, serving inference requests for a simple image recognition API. Go back to the [Serve a Model tutorial](../README.md#serve-a-model) and try to run this model archive that you just created!


### Model specific custom python requirements
Custom models/handlers may depend on different python packages which are not installed by-default as a part of `TorchServe` setup.
Supply a [python requirements](https://pip.pypa.io/en/stable/user_guide/#requirements-files) file containing the list of required python packages to be installed by `TorchServe` for seamless model serving using `--requirements-file` parameter while creating the model-archiver.

Example:
```bash
torch-model-archiver --model-name densenet_161 --version 1.0 --model-file model.py --serialized-file model.pt --handler image_classifier --requirements-file <path_to_custom_requirements_file>
```

**Note**: This feature is by-default disabled in TorchServe and needs to be enabled through configuration. For more details refer [TorchServe's configuration documentation](../docs/configuration.md#allow-model-specific-custom-python-packages)
