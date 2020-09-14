# TorchServe release benchmarking

This document provides the details for execution of automated Performance Benchmark Tests for TorchServe release. 
These tests aim to emulate the setup described in the baseline document

## Apache Bench Tests:

### Test Setup

 - Instance Type
    - CPU : c4.4xlarge
    - GPU : p3.8xlarge

 - Workers : 4
 - Apache Bench tool - Benchmark script from the TS benchmarks dir 
 - Number of Requests: 10,000
 - Batch Delay: 50ms
 - Batch Size: 1/2/4/8
 
### Models used

 - FasterRCNN (Object Detector) :
   - Model : Eager
   - Environment : CPU/GPU
   - Mar file : https://torchserve.s3.amazonaws.com/mar_files/fastrcnn.mar
   - Input : https://github.com/pytorch/serve/blob/master/examples/object_detector/persons.jpg

 - VGG11:
   - Model : Eager
   - Environment : CPU/GPU
   - Mar file : https://torchserve.s3.amazonaws.com/mar_files/vgg11_v2.mar
   - Input : https://github.com/pytorch/serve/blob/master/examples/image_classifier/kitten.jpg
   
 - VGG11:
   - Mode : Scripted
   - Environment : CPU/GPU
   - Mar file : https://torchserve.s3.amazonaws.com/mar_files/vgg11_scripted.mar
   - Input : https://github.com/pytorch/serve/blob/master/examples/image_classifier/kitten.jpg
   
 - Bert:
   - Mode : Scripted
   - Environment : GPU
   - Mar file : https://torchserve.s3.amazonaws.com/mar_files/BERTSeqClassification_Torchscript_gpu.mar
   - Input : https://github.com/pytorch/serve/blob/master/examples/Huggingface_Transformers/Seq_classification_artifacts/sample_text0.txt
 
 - Bert:
   - Mode : Scripted
   - Environment : CPU
   - Mar file : https://torchserve.s3.amazonaws.com/mar_files/BERTSeqClassification_Torchscript_batch.mar
   - Input : https://github.com/pytorch/serve/blob/master/examples/Huggingface_Transformers/Seq_classification_artifacts/sample_text0.txt

## Release Benchmark Script

The [script](run_release_benchmark.py) aims to automate the benchmarking activities for TorchServe release. The script automates the following :

    - Start EC2 instance based on instance type : CPU (c4.4xlarge), GPU (p3.8xlarge)
    - Run benchmark based on model (vgg11/fasterrcnn/bert) and mode (scripted/eager)
    - Collect benchmark results
    - Terminate the EC2 instance

## Pre-requisites

### Install pip dependencies

`pip install -U -r release_benchmark-requirements.txt`

### Configure AWS access key & secrete key

Refer the [AWS documentation](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-files.html) to setup the AWS credentials required for accessing AWS EC2 service.

## Run benchmark

## Benchmark parameters

|Parameter|Required|Descripted|Default|
|---|---|---|---|
|subnet_id|:heavy_check_mark:|Id of the subnet to be used while creating EC2 instance||
|security_group_id|:heavy_check_mark:|Id of the securtiy group to be used while creating EC2 instance||
|model_name||Name of the model to run benchmark on. Possible values - vgg11/fastrcnn/bert|vgg11|
|model_mode||Mode of the model. Possible values: scripted/eager|eager|
|batch_size||Batch size of the model.|1|
|instance_type||EC2 instance type to be used. Possible values : cpu/gpu|cpu|
|branch||Branch name which needs to be benchmarked|master|
|ami||AWS ami id to use while creating EC2 instance|ami-079d181e97ab77906|

### Benchmark reports
The CSV report are generated at provide s3 location.

### Sample output CSV
| Benchmark | Model | Concurrency | Requests | TS failed requests | TS throughput | TS latency P50 | TS latency P90| TS latency P90 | TS latency mean | TS error rate | Model_p50 | Model_p90 | Model_p99 |
|---|---|---|---|---|---|---|---|---|---|---|---|---| ---|
| AB | https://torchserve.s3.amazonaws.com/mar_files/squeezenet1_1.mar | 10 | 100 | 0 | 15.66 | 512 | 1191 | 2024 | 638.695 | 0 | 196.57 | 270.9 | 106.53|
