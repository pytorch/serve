# Torch Model archiver for TS

## Contents of this Document
* [Overview](#overview)
* [Torch Model Archiver CLI](#torch-model-archiver-command-line-interface)
* [Artifact Details](#artifact-details)
    * [MAR-INFO](#mar-inf)
    * [Model name](#model-name)
    * [Runtime](#runtime)
    * [Handler](#handler)
* [Quick Start: Creating a Model Archive](#creating-a-model-archive)

## Overview

A key feature of TS is the ability to package all model artifacts into a single model archive file. It is a separate command line interface (CLI), `torch-model-archiver`, that can take model checkpoints or model definition file with state_dict, and package them into a `.mar` file. This file can then be redistributed and served by anyone using TS. It takes in the following model artifacts: a model checkpoint file in case of torchscript or a model definition file and a state_dict file in case of eager mode, and other optional assets that may be required to serve the model. The CLI creates a `.mar` file that TS's server CLI uses to serve the models.

**Important**: Make sure you try the [Quick Start: Creating a Model Archive](#creating-a-model-archive) tutorial for a short example of using `torch-model-archiver`.

The following information is required to create a standalone model archive:
1. [Model name](#model-name)
2. [Model file](#model-file)
2. [Serialized file](#serialized-file)

## Torch Model Archiver Command Line Interface

Now let's cover the details on using the CLI tool: `model-archiver`.

Here is an example usage with the squeezenet_v1.1 model archive following the example in the [examples README](../examples/README.md):

```bash

torch-model-archiver --model-name densenet161 --model-file serve/examples/densenet_161/model.py --serialized-file densenet161-8d451a50.pth --extra-files serve/examples/index_to_name.json

```

### Arguments

```
$ model-archiver -h
usage: torch-model-archiver [-h] --model-name MODEL_NAME --model-path MODEL_PATH
                      --handler HANDLER [--runtime {python,python2,python3}]
                      [--export-path EXPORT_PATH] [-f]

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
                        class definition extended from torch.nn.modules.
  --handler HANDLER     Handler path to handle custom TS inference logic.
  --extra-files EXTRA_FILES
                        Comma separated path to extra dependency files.
  --runtime {python,python2,python3}
                        The runtime specifies which language to run your
                        inference code on. The default runtime is
                        RuntimeType.PYTHON. At the present moment we support
                        the following runtimes python, python2, python3
  --export-path EXPORT_PATH
                        Path where the exported .mar file will be saved. This
                        is an optional parameter. If --export-path is not
                        specified, the file will be saved in the current
                        working directory.
  --archive-format {tgz,default}
                        The format in which the model artifacts are archived.
                        "tgz": This creates the model-archive in <model-name>.tar.gz format.
                        If platform hosting MMS requires model-artifacts to be in ".tar.gz"
                        use this option.
                        "no-archive": This option creates an non-archived version of model artifacts
                        at "export-path/{model-name}" location. As a result of this choice,
                        MANIFEST file will be created at "export-path/{model-name}" location
                        without archiving these model files
                        "default": This creates the model-archive in <model-name>.mar format.
                        This is the default archiving format. Models archived in this format
                        will be readily hostable on native MMS.
  -f, --force           When the -f or --force flag is specified, an existing
                        .mar file with same name as that provided in --model-
                        name in the path specified by --export-path will
                        overwritten
```

## Artifact Details

### MAR-INF
**MAR-INF** is a reserved folder name that will be used inside `.mar` file. This folder contains the model archive metadata files. Users should avoid using **MAR-INF** in their model path.

### Runtime

### Model name

A valid model name must begin with a letter of the alphabet and can only contains letters, digits, underscores (_), dashes (-) and periods (.).

**Note**: The model name can be overridden when you register the model with [Register Model API](../docs/management_api.md#register-a-model).

### Model file

A model file should contain the model architecture. This file is mandatory in case of eager mode models.

### Serialized file

A serialized file (.pt or .pth) should be a checkpoint in case of torchscript and state_dict in case of eager mode.


## Creating a Model Archive

**1. Download these sample Densenet model artifacts (if you don't have them handy)**

```bash
TODO : add sample artifact URLs
```

The downloaded model artifact files are:

TODO : add description about the above artifacts


**2. Download the torch model archiver source**
```bash
git clone https://github.com/pytorch/serve.git
```

**3. Prepare your model custom service code**

You can implement your own model customer service code with a model archive entry point.
Here we are going to use the MXNet vision service `model_service_template`.
This template is one of several provided with MMS.
Download the template and place it in your `squeezenet` folder.

```bash
cp -r mxnet-model-server/examples/model_service_template/* squeezenet/
```

**4. Package your model**

With the model artifacts available locally, you can use the `torch-model-archiver` CLI to generate a `.mar` file that can be used to serve an inference API with MMS.

In this next step we'll run `torch-model-archiver` and tell it our model's name is `densenet_161` with the `model-name` argument. Then we're giving it the `model-file` and `serialized-file` to the model's assets.

For torchscript:
```bash
torch-model-archiver --model-name densenet_161 --serialized-file model.pt
```

For eagermode:
```bash
torch-model-archiver --model-name densenet_161 --model-flie model.py --serialized-file model.pt
```

This will package all the model artifacts files and output `squeezenet_v1.1.mar` in the current working directory. This `.mar` file is all you need to run TS, serving inference requests for a simple image recognition API. Go back to the [Serve a Model tutorial](../README.md#serve-a-model) and try to run this model archive that you just created!
