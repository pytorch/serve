# M1 support

## M1 Support 
TorchServe supports Mac OS with M1 hardware. 

1. TorchServe CI jobs now include M1 hardware in order to ensure support, [documentation](https://docs.github.com/en/actions/using-github-hosted-runners/about-github-hosted-runners/about-github-hosted-runners#standard-github-hosted-runners-for-public-repositories) on github M1 hardware.
    - [Regression Tests](https://github.com/pytorch/serve/blob/master/.github/workflows/regression_tests_cpu.yml)  
    - [Regression binaries Test](https://github.com/pytorch/serve/blob/master/.github/workflows/regression_tests_cpu_binaries.yml) 
2. For [Docker](https://docs.docker.com/desktop/install/mac-install/) ensure Docker for Apple silicon is installed then follow [setup steps](https://github.com/pytorch/serve/tree/master/docker)
## Running Torchserve on M1
 
Follow [getting started documentation](https://github.com/pytorch/serve?tab=readme-ov-file#-quick-start-with-torchserve-conda) 

### Example
 
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
Model server started.
```
