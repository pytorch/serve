# Torch Model archiver for TS

## Contents of this Document
* [Overview](#overview)
* [Model Archiver CLI](#model-archiver-command-line-interface)
* [Artifact Details](#artifact-details)
    * [MAR-INFO](#mar-inf)
    * [Model name](#model-name)
    * [Runtime](#runtime)
    * [Handler](#handler)
* [Quick Start: Creating a Model Archive](#creating-a-model-archive)

## Overview

A key feature of TS is the ability to package all model artifacts into a single model archive file. It is a separate command line interface (CLI), `torch-model-archiver`, that can take model checkpoints and package them into a `.mar` file. This file can then be redistributed and served by anyone using TS. It takes in the following model artifacts: a model composed of one or more files, the description of the model's inputs in the form of a signature file, a service file describing how to handle inputs and outputs, and other optional assets that may be required to serve the model. The CLI creates a `.mar` file that TS's server CLI uses to serve the models.

**Important**: Make sure you try the [Quick Start: Creating a Model Archive](#creating-a-model-archive) tutorial for a short example of using `torch-model-archiver`.

TS can support any arbitrary model file. It is the custom service code's responsibility to locate and load the model files. The following information is required to create a standalone model archive:
1. [Model name](#model-name)
2. [Model path](#model-path)
3. [Handler](#handler)

## Torch Model Archiver Command Line Interface

Now let's cover the details on using the CLI tool: `model-archiver`.

Here is an example usage with the squeezenet_v1.1 model archive which you can download or create by following the example in the [main README](../README.md):

```bash

torch-model-archiver --model-name squeezenet_v1.1 --model-path squeezenet --handler mxnet_vision_service:handle

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
  --model-path MODEL_PATH
                        Path to the folder containing model related files.
  --handler HANDLER     Handler path to handle custom MMS inference logic.
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

### Model path

A folder that contains all necessary files needed to run inference code for the model. All the files and sub-folders (except [excluded files](#excluded-files)) will be packaged into the `.mar` file.

#### excluded files
The following types of file will be excluded during model archive packaging:
1. hidden files
2. Mac system files: __MACOSX and .DS_Store
3. MANIFEST.json
4. python compiled byte code (.pyc) files and cache folder __pycache__

### handler

A handler is a python entry point that MMS can invoke to execute inference code. The format of a Python handler is:
* python_module_name[:function_name] (for example: lstm-service:handle).

The function name is optional if the provided python module follows one of predefined conventions:
1. There is a `handle()` function available in the module
2. The module contains only one Class and that class contains a `handle()` function.

Further details and specifications are found on the [custom service](../docs/custom_service.md) page.


## Creating a Model Archive

**1. Download these sample SqueezeNet model artifacts (if you don't have them handy)**

```bash
mkdir squeezenet

curl -o squeezenet https://s3.amazonaws.com/model-server/model_archive_1.0/examples/squeezenet_v1.1/squeezenet_v1.1-symbol.json
curl -o squeezenet https://s3.amazonaws.com/model-server/model_archive_1.0/examples/squeezenet_v1.1/squeezenet_v1.1-0000.params
curl -o squeezenet https://s3.amazonaws.com/model-server/model_archive_1.0/examples/squeezenet_v1.1/signature.json
curl -o squeezenet https://s3.amazonaws.com/model-server/model_archive_1.0/examples/squeezenet_v1.1/synset.txt
```

The downloaded model artifact files are:

* **Model Definition** (json file) - contains the layers and overall structure of the neural network.
* **Model Params and Weights** (params file) - contains the parameters and the weights.
* **Model Signature** (json file) - defines the inputs and outputs that MMS is expecting to hand-off to the API.
* **assets** (text files) - auxiliary files that support model inference such as vocabularies, labels, etc. These vary depending on the model.


**2. Download the model archiver source**
```bash
git clone https://github.com/awslabs/mxnet-model-server.git
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

With the model artifacts available locally, you can use the `model-archiver` CLI to generate a `.mar` file that can be used to serve an inference API with MMS.

In this next step we'll run `model-archiver` and tell it our model's prefix is `squeezenet_v1.1` with the `model-name` argument. Then we're giving it the `model-path` to the model's assets.

**Note**: For mxnet models, `model-name` must match prefix of the symbol and param file name.

```bash
model-archiver --model-name squeezenet_v1.1 --model-path squeezenet --handler mxnet_vision_service:handle
```

This will package all the model artifacts files located in the `squeezenet` directory and output `squeezenet_v1.1.mar` in the current working directory. This `.mar` file is all you need to run MMS, serving inference requests for a simple image recognition API. Go back to the [Serve a Model tutorial](../README.md#serve-a-model) and try to run this model archive that you just created!
