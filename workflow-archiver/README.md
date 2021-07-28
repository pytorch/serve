# Torch Workflow archiver for TorchServe

## Contents of this Document
* [Overview](#overview)
* [Installation](#installation)
* [Torch Workflow Archiver CLI](#torch-workflow-archiver-command-line-interface)
* [Artifact Details](#artifact-details)
    * [WAR-INFO](#war-inf)
    * [Workflow name](#workflow-name)
    * [Spec File](#spec-file)
    * [handler](#handler)

## Overview

A key feature of TorchServe is the ability to package workflow specification (.yaml) and other workflow dependency files into a single workflow archive file (.war). This file can then be redistributed and served by anyone using TorchServe.
 
The CLI creates a `.war` file that TorchServe CLI uses to serve the workflows.

The following information is required to create a standalone workflow archive:
1. [Workflow name](#workflow-name)
2. [Spec file](#spec-file)

## Installation

Install `torch-workflow-archiver` as follows:

```bash
pip install torch-workflow-archiver
```

## Installation from source

Install `torch-workflow-archiver` from source as follows:

```bash
git clone https://github.com/pytorch/serve.git
cd serve/workflow-archiver
pip install .
```

## Torch Workflow Archiver Command Line Interface

Now let's cover the details on using the CLI tool: `torch-workflow-archiver`.

```bash
torch-workflow-archiver --workflow-name my_workflow --spec-file spec.yaml --handler handler.py
```

### Arguments

```
$ torch-workflow-archiver -h
usage: torch-workflow-archiver [-h] --workflow-name WORKFLOW_NAME --spec-file WORKFLOW_SPECIFICATION_FILE_PATH
                      [--handler HANDLER] [--export-path EXPORT_PATH] [-f]

Workflow Archiver Tool

optional arguments:
  -h, --help            show this help message and exit
  --workflow-name WORKFLOW_NAME
                        Exported workflow name. Exported file will be named as
                        workflow-name.war and saved in current working directory
                        if no --export-path is specified, else it will be
                        saved under the export path
  --spec-file WORKFLOW_SPECIFICATION_FILE_PATH
                        Path to .yaml file containing workflow DAG specification.
  --handler HANDLER     Path to python file containing workflow's pre-process and post-process logic.
  --export-path EXPORT_PATH
                        Path where the exported .war file will be saved. This
                        is an optional parameter. If --export-path is not
                        specified, the file will be saved in the current
                        working directory.
  -f, --force           When the -f or --force flag is specified, an existing
                        .war file with same name as that provided in --workflow-name
                        in the path specified by --export-path will be overwritten
  --extra-files EXTRA_FILES
                        Comma separated path to extra dependency files.
```

## Artifact Details

### WAR-INF
**WAR-INF** is a reserved folder name that will be used inside `.war` file. This folder contains the workflow archive metadata files. Users should avoid using **WAR-INF** in their workflow path.

### Runtime

### Workflow name

A valid workflow name must begin with a letter of the alphabet and can only contains letters, digits, underscores `_`, dashes `-` and periods `.`.

### Spec file

A .yaml file specifying workflow DAG specification

### Handler

Handler is path to a py file to handle workflow's pre-process and post-process functions. 
