# Apple Silicon Support 

## What is supported 
* TorchServe CI jobs now include M1 hardware in order to ensure support, [documentation](https://docs.github.com/en/actions/using-github-hosted-runners/about-github-hosted-runners/about-github-hosted-runners#standard-github-hosted-runners-for-public-repositories) on github M1 hardware.
    - [Regression Tests](https://github.com/pytorch/serve/blob/master/.github/workflows/regression_tests_cpu.yml)  
    - [Regression binaries Test](https://github.com/pytorch/serve/blob/master/.github/workflows/regression_tests_cpu_binaries.yml) 
* For [Docker](https://docs.docker.com/desktop/install/mac-install/) ensure Docker for Apple silicon is installed then follow [setup steps](https://github.com/pytorch/serve/tree/master/docker)

## Experimental Support

* For GPU jobs on Apple Silicon, [MPS](https://pytorch.org/docs/master/notes/mps.html) is now auto detected and enabled. To prevent TorchServe from using MPS, users have to set `deviceType: "cpu"` in model-config.yaml. 
    * This is an experimental feature and NOT ALL models are guaranteed to work.  
* Number of GPUs now reports GPUs on Apple Silicon

### Testing 
* [Pytests](https://github.com/pytorch/serve/tree/master/test/pytest/test_device_config.py) that checks for MPS on MacOS M1 devices 
* Models that have been tested and work: Resnet-18, Densenet161, Alexnet
* Models that have been tested and DO NOT work: MNIST


#### Example Resnet-18 Using MPS On Mac M1 Pro
```
serve % torchserve --start --model-store model_store_gen --models resnet-18=resnet-18.mar --ncs

Torchserve version: 0.10.0
Number of GPUs: 16
Number of CPUs: 10
Max heap size: 8192 M
Python executable: /Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11
Config file: N/A
Inference address: http://127.0.0.1:8080
Management address: http://127.0.0.1:8081
Metrics address: http://127.0.0.1:8082
Model Store: 
Initial Models: resnet-18=resnet-18.mar
Log dir: 
Metrics dir: 
Netty threads: 0
Netty client threads: 0
Default workers per model: 16
Blacklist Regex: N/A
Maximum Response Size: 6553500
Maximum Request Size: 6553500
Limit Maximum Image Pixels: true
Prefer direct buffer: false
Allowed Urls: [file://.*|http(s)?://.*]
Custom python dependency for model allowed: false
Enable metrics API: true
Metrics mode: LOG
Disable system metrics: false
Workflow Store: 
CPP log config: N/A
Model config: N/A
024-04-08T14:18:02,380 [INFO ] main org.pytorch.serve.servingsdk.impl.PluginsManager -  Loading snapshot serializer plugin...
2024-04-08T14:18:02,391 [INFO ] main org.pytorch.serve.ModelServer - Loading initial models: resnet-18.mar
2024-04-08T14:18:02,699 [DEBUG] main org.pytorch.serve.wlm.ModelVersionedRefs - Adding new version 1.0 for model resnet-18
2024-04-08T14:18:02,699 [INFO ] main org.pytorch.serve.wlm.ModelManager - Model resnet-18 loaded.
2024-04-08T14:18:02,699 [DEBUG] main org.pytorch.serve.wlm.ModelManager - updateModel: resnet-18, count: 16
...
...
serve % curl http://127.0.0.1:8080/predictions/resnet-18 -T ./examples/image_classifier/kitten.jpg
...
{
  "tabby": 0.40966302156448364,
  "tiger_cat": 0.3467046618461609,
  "Egyptian_cat": 0.1300288736820221,
  "lynx": 0.02391958422958851,
  "bucket": 0.011532187461853027
}
...
```
#### Conda Example 

```
(myenv) serve % pip list | grep torch                                                                   
torch                     2.2.1
torchaudio                2.2.1
torchdata                 0.7.1
torchtext                 0.17.1
torchvision               0.17.1
(myenv3) serve % conda install -c pytorch-nightly torchserve torch-model-archiver torch-workflow-archiver
(myenv3) serve % pip list | grep torch                                                                   
torch                     2.2.1
torch-model-archiver      0.10.0b20240312
torch-workflow-archiver   0.2.12b20240312
torchaudio                2.2.1
torchdata                 0.7.1
torchserve                0.10.0b20240312
torchtext                 0.17.1
torchvision               0.17.1
(myenv3) serve % torchserve --start --ncs  --models densenet161.mar --model-store ./model_store_gen/
Torchserve version: 0.10.0
Number of GPUs: 0
Number of CPUs: 10
Max heap size: 8192 M
Config file: N/A
Inference address: http://127.0.0.1:8080
Management address: http://127.0.0.1:8081
Metrics address: http://127.0.0.1:8082
Initial Models: densenet161.mar
Netty threads: 0
Netty client threads: 0
Default workers per model: 10
Blacklist Regex: N/A
Maximum Response Size: 6553500
Maximum Request Size: 6553500
Limit Maximum Image Pixels: true
Prefer direct buffer: false
Allowed Urls: [file://.*|http(s)?://.*]
Custom python dependency for model allowed: false
Enable metrics API: true
Metrics mode: LOG
Disable system metrics: false
CPP log config: N/A
Model config: N/A
System metrics command: default
...
2024-03-12T15:58:54,702 [INFO ] main org.pytorch.serve.wlm.ModelManager - Model densenet161 loaded.
2024-03-12T15:58:54,702 [DEBUG] main org.pytorch.serve.wlm.ModelManager - updateModel: densenet161, count: 10
Model server started.
...
(myenv3) serve % curl http://127.0.0.1:8080/predictions/densenet161 -T examples/image_classifier/kitten.jpg 
{
  "tabby": 0.46661922335624695,
  "tiger_cat": 0.46449029445648193,
  "Egyptian_cat": 0.0661405548453331,
  "lynx": 0.001292439759708941,
  "plastic_bag": 0.00022909720428287983
}