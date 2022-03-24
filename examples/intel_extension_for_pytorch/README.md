# TorchServe with Intel® Extension for PyTorch*

TorchServe can be used with Intel® Extension for PyTorch* (IPEX) to give performance boost on Intel hardware<sup>1</sup>. 
Here we show how to use TorchServe with IPEX.

<sup>1. While IPEX benefits all platforms, platforms with AVX512 benefit the most. </sup>

## Contents of this Document 
* [Install Intel Extension for PyTorch](#install-intel-extension-for-pytorch)
* [Serving model with Intel Extension for PyTorch](#serving-model-with-intel-extension-for-pytorch)
* [TorchServe with Launcher](#torchserve-with-launcher)
* [Creating and Exporting INT8 model for IPEX](#creating-and-exporting-int8-model-for-ipex)
* [Boosting Performance of TorchServe Multi Worker Inference with Launcher Core Pinning](#boosting-performance-of-torchserve-multi-worker-inference-with-launcher-core-pinning)
* [Benchmarking with Launcher](#benchmarking-with-launcher)
* [Performance Boost with IPEX and Launcher](#performance-boost-with-ipex-and-launcher)


## Install Intel Extension for PyTorch 
Refer to the documentation [here](https://github.com/intel/intel-extension-for-pytorch#installation). 

## Serving model with Intel Extension for PyTorch  
After installation, all it needs to be done to use TorchServe with IPEX is to enable it in `config.properties`. 
```
ipex_enable=true
```
Once IPEX is enabled, deploying PyTorch model follows the same procedure shown [here](https://pytorch.org/serve/use_cases.html). TorchServe with IPEX can deploy any model and do inference. 

## TorchServe with Launcher
Launcher is a script to automate the process of tunining configuration setting on intel hardware to boost performance. Tuning configurations such as OMP_NUM_THREADS, thread affininty, memory allocator can have a dramatic effect on performance. Please refer to [here](https://github.com/intel/intel-extension-for-pytorch/blob/master/docs/tutorials/performance_tuning/tuning_guide.md) and [here](https://github.com/intel/intel-extension-for-pytorch/blob/master/docs/tutorials/performance_tuning/launch_script.md) for details on performance tuning with launcher. 

All it needs to be done to use TorchServe with launcher is to set its configuration in `config.properties`.

Add the following lines in `config.properties` to use launcher with its default configuration. 
```
ipex_enable=true
cpu_launcher_enable=true
```

Launcher by default uses `numactl` if its installed to ensure socket is pinned and thus memory is allocated from local numa node. To use launcher without numactl, add the following lines in `config.properties`.
```
ipex_enable=true
cpu_launcher_enable=true
cpu_launcher_args=--disable_numactl
```

Launcher by default uses only non-hyperthreaded cores if hyperthreading is present to avoid core compute resource sharing. To use launcher with all cores, both physical and logical, add the following lines in `config.properties`.  
```
ipex_enable=true
cpu_launcher_enable=true
cpu_launcher_args=--use_logical_core
```

Below is an example of passing multiple args to `cpu_launcher_args`.
```
ipex_enable=true
cpu_launcher_enable=true
cpu_launcher_args=--use_logical_core --disable_numactl 
```

Below is some useful `cpu_launcher_args` to note. Italic values are default if applicable.
1. Memory Allocator: [ PTMalloc `--use_default_allocator` | *TCMalloc `--enable_tcmalloc`* | JeMalloc `--enable_jemalloc`]
   * PyTorch by defualt uses PTMalloc. TCMalloc/JeMalloc generally gives better performance.
2. OpenMP library: [GNU OpenMP `--disable_iomp` | *Intel OpenMP*]
   * PyTorch by default uses GNU OpenMP. Launcher by default uses Intel OpenMP. Intel OpenMP library generally gives better performance.
3. Node id: [`--node_id`]
   * Launcher by default uses all NUMA nodes. Limit memory access to local memories on the Nth Numa node to avoid Non-Uniform Memory Access (NUMA).

Please refer to [here](https://github.com/intel/intel-extension-for-pytorch/blob/master/docs/tutorials/performance_tuning/launch_script.md) for a full list of tunable configuration of launcher. 

Please refer to [here](https://github.com/intel/intel-extension-for-pytorch/blob/master/docs/tutorials/performance_tuning/launch_script.md) for more details. 

## Creating and Exporting INT8 model for IPEX
Intel Extension for PyTorch supports both eager and torchscript mode. In this section, we show how to deploy INT8 model for IPEX. 

### 1. Creating a serialized file 
First create `.pt` serialized file using IPEX INT8 inference. Here we show two examples with BERT and ResNet50. 

#### BERT

```
import torch
import intel_extension_for_pytorch as ipex
import transformers
from transformers import AutoModelForSequenceClassification, AutoConfig

# load the model 
config = AutoConfig.from_pretrained(
    "bert-base-uncased", return_dict=False, torchscript=True, num_labels=2)
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", config=config)
model = model.eval()

# define dummy input tensor to use for the model's forward call to record operations in the model for tracing
N, max_length = 1, 384 
dummy_tensor = torch.ones((N, max_length), dtype=torch.long)

# calibration 
# ipex supports two quantization schemes to be used for activation: torch.per_tensor_affine and torch.per_tensor_symmetric
# default qscheme is torch.per_tensor_affine
conf = ipex.quantization.QuantConf(qscheme=torch.per_tensor_affine)
n_iter = 100
with torch.no_grad():
    for i in range(n_iter):
        with ipex.quantization.calibrate(conf):
            model(dummy_tensor, dummy_tensor, dummy_tensor)

# optionally save the configuraiton for later use
# save:
# conf.save("model_conf.json")
# load:
# conf = ipex.quantization.QuantConf("model_conf.json")

# conversion 
jit_inputs = (dummy_tensor, dummy_tensor, dummy_tensor)
model = ipex.quantization.convert(model, conf, jit_inputs)

# enable fusion path work(need to run forward propagation twice)
with torch.no_grad():
    y = model(dummy_tensor,dummy_tensor,dummy_tensor)
    y = model(dummy_tensor,dummy_tensor,dummy_tensor)

# save to .pt 
torch.jit.save(model, 'bert_int8_jit.pt')
```

#### ResNet50 

```
import torch
import torch.fx.experimental.optimization as optimization
import intel_extension_for_pytorch as ipex
import torchvision.models as models

# load the model
model = models.resnet50(pretrained=True)
model = model.eval()
model = optimization.fuse(model)

# define dummy input tensor to use for the model's forward call to record operations in the model for tracing
N, C, H, W = 1, 3, 224, 224
dummy_tensor = torch.randn(N, C, H, W).contiguous(memory_format=torch.channels_last)

# calibration
# ipex supports two quantization schemes to be used for activation: torch.per_tensor_affine and torch.per_tensor_symmetric
# default qscheme is torch.per_tensor_affine
conf = ipex.quantization.QuantConf(qscheme=torch.per_tensor_symmetric)
n_iter = 100
with torch.no_grad():
    for i in range(n_iter):
        with ipex.quantization.calibrate(conf):
           model(dummy_tensor)

# optionally save the configuraiton for later use
# save:
# conf.save("model_conf.json")
# load:
# conf = ipex.quantization.QuantConf("model_conf.json")

# conversion
jit_inputs = (dummy_tensor)
model = ipex.quantization.convert(model, conf, jit_inputs)

# enable fusion path work(need to run two iterations)
with torch.no_grad():
    y = model(dummy_tensor)
    y = model(dummy_tensor)

# save to .pt
torch.jit.save(model, 'rn50_int8_jit.pt')
```

### 2. Creating a Model Archive 
Once the serialized file ( `.pt`) is created, it can be used with `torch-model-archiver` as ususal. Use the following command to package the model.  
```
torch-model-archiver --model-name rn50_ipex_int8 --version 1.0 --serialized-file rn50_int8_jit.pt --handler image_classifier 
```
### 3. Start TorchServe to serve the model 
Make sure to set `ipex_enable=true` in `config.properties`. Use the following command to start TorchServe with IPEX. 
```
torchserve --start --ncs --model-store model_store --ts-config config.properties
```

### 4. Registering and Deploying model 
Registering and deploying the model follows the same steps shown [here](https://pytorch.org/serve/use_cases.html). 

## Boosting Performance of TorchServe Multi Worker Inference with Launcher Core Pinning
When running [multi-worker inference](https://pytorch.org/serve/management_api.html#scale-workers) with Torchserve, launcher pin cores to workers to boost performance. Internally, launcher equally divides the number of cores by the number of workers such that each worker is pinned to assigned cores. Doing so avoids core overlap between workers which can signficantly boost performance for TorchServe multi-worker inference.

We'll use TorchServe official [benchmark](https://github.com/pytorch/serve/tree/master/benchmarks) to demonstrate this, but keep in mind that this is a generic feature applicable to any TorchServe multi-worker inference use casese. 

For example, assume running 4 workers 
```
python benchmark-ab.py --workers 4
```
on a machine with Intel(R) Xeon(R) Platinum 8180 CPU, 2 sockets, 28 cores per socket, 2 threads per core. Launcher will bind worker 0 to cores 0-13, worker 1 to cores 14-27, worker 2 to cores 28-41, and worker 3 to cores 42-55. 

All it needs to be done to use TorchServe with launcher's core pinning is to enable launcher in `config.properties`.

Add the following lines to `config.properties` in the benchmark directory to use launcher's core pinning:
```
cpu_launcher_enable=true
```

CPU usage is shown as below:
![launcher_core_pinning](https://user-images.githubusercontent.com/93151422/159063975-e7e8d4b0-e083-4733-bdb6-4d92bdc10556.gif)


4 main worker threads were launched, then each launched a num_physical_cores/num_workers number (14) of threads affinitized to the assigned physical cores. 

#### Scale workers
Additionally when dynamically [scaling workers](https://pytorch.org/serve/management_api.html#scale-workers), cores that were pinned to killed workers by the launcher could be left unutilized. To address this problem, launcher internally restarts the workers to re-distribute cores that were pinned to killed workers to the remaining, alive workers. This is taken care internally, so users do not have to worry about this. 

For example, let us continue with the above example with 4 workers - binding worker 0 to cores 0-13, worker 1 to cores 14-27, worker 2 to cores 28-41, and worker 3 to cores 42-55. Assume killing workers 2 and 3. If cores were not re-distributed after the scale down, cores 28-55 would be left unutilized. Instead, launcher re-distributes cores 28-55 to workers 0 and 1 such that now worker 0 binds to cores 0-27 and worker 1 binds to cores 28-55.

## Benchmarking with Launcher 
Launcher can be used with TorchServe official [benchmark](https://github.com/pytorch/serve/tree/master/benchmarks) to launch server and benchmark requests with optimal configuration on Intel hardware.

In this section we provide examples of benchmarking with launcher with its default configuration.

Add the following lines to `config.properties` in the benchmark directory to use launcher with its default setting. 
```
ipex_enable=true
cpu_launcher_enable=true
```

The rest of the steps for benchmarking follows the same steps shown [here](https://github.com/pytorch/serve/tree/master/benchmarks).

`model_log.log` contains information and command that were used for this execution launch. 


CPU usage on a machine with Intel(R) Xeon(R) Platinum 8180 CPU, 2 sockets, 28 cores per socket, 2 threads per core is shown as below: 
![launcher_default_2sockets](https://user-images.githubusercontent.com/93151422/144373537-07787510-039d-44c4-8cfd-6afeeb64ac78.gif)

```
$ cat logs/model_log.log
2021-12-01 21:22:40,096 - __main__ - WARNING - Both TCMalloc and JeMalloc are not found in $CONDA_PREFIX/lib or $VIRTUAL_ENV/lib or /.local/lib/ or /usr/local/lib/ or /usr/local/lib64/ or /usr/lib or /usr/lib64 or /home/<user>/.local/lib/ so the LD_PRELOAD environment variable will not be set. This may drop the performance
2021-12-01 21:22:40,096 - __main__ - INFO - OMP_NUM_THREADS=56
2021-12-01 21:22:40,096 - __main__ - INFO - Using Intel OpenMP
2021-12-01 21:22:40,096 - __main__ - INFO - KMP_AFFINITY=granularity=fine,compact,1,0
2021-12-01 21:22:40,096 - __main__ - INFO - KMP_BLOCKTIME=1
2021-12-01 21:22:40,096 - __main__ - INFO - LD_PRELOAD=<VIRTUAL_ENV>/lib/libiomp5.so
2021-12-01 21:22:40,096 - __main__ - WARNING - Numa Aware: cores:[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55] in different NUMA node
```

CPU usage on a machine with Intel(R) Xeon(R) Platinum 8375C CPU, 1 socket, 2 cores per socket, 2 threads per socket is shown as below: 
![launcher_default_1socket](https://user-images.githubusercontent.com/93151422/144372993-92b2ca96-f309-41e2-a5c8-bf2143815c93.gif)

```
$ cat logs/model_log.log
2021-12-02 06:15:03,981 - __main__ - WARNING - Both TCMalloc and JeMalloc are not found in $CONDA_PREFIX/lib or $VIRTUAL_ENV/lib or /.local/lib/ or /usr/local/lib/ or /usr/local/lib64/ or /usr/lib or /usr/lib64 or /home/<user>/.local/lib/ so the LD_PRELOAD environment variable will not be set. This may drop the performance
2021-12-02 06:15:03,981 - __main__ - INFO - OMP_NUM_THREADS=2
2021-12-02 06:15:03,982 - __main__ - INFO - Using Intel OpenMP
2021-12-02 06:15:03,982 - __main__ - INFO - KMP_AFFINITY=granularity=fine,compact,1,0
2021-12-02 06:15:03,982 - __main__ - INFO - KMP_BLOCKTIME=1
2021-12-02 06:15:03,982 - __main__ - INFO - LD_PRELOAD=<VIRTUAL_ENV>/lib/libiomp5.so

```

## Performance Boost with IPEX and Launcher

![pdt_perf](https://user-images.githubusercontent.com/93151422/159067306-dfd604e3-8c66-4365-91ae-c99f68d972d5.png)


Above shows performance improvement of Torchserve with IPEX and launcher on ResNet50 and BERT-base-uncased. Torchserve official [apache-bench benchmark](https://github.com/pytorch/serve/tree/master/benchmarks#benchmarking-with-apache-bench) on Amazon EC2 m6i.24xlarge was used to collect the results<sup>2</sup>. Add the following lines in ```config.properties``` to reproduce the results. Notice that launcher is configured such that a single instance uses all physical cores on a single socket to avoid cross socket communication and core overlap. 

```
ipex_enable=true
cpu_launcher_enable=true
cpu_launcher_args=--node_id 0 --enable_jemalloc
```
Use the following command to reproduce the results. 
```
python benchmark-ab.py --url {modelUrl} --input {inputPath} --concurrency 1 
```

For example, run the following command to reproduce latency performance of ResNet50 with data type of IPEX int8 and batch size of 1. 
```
python benchmark-ab.py --url 'file:///model_store/rn50_ipex_int8.mar' --concurrency 1
```

For example, run the following command to reproduce latency performance of BERT with data type of IPEX int8 and batch size of 1. 
```
python benchmark-ab.py --url 'file:///model_store/bert_ipex_int8.mar' --input '../examples/Huggingface_Transformers/Seq_classification_artifacts/sample_text_captum_input.txt' --concurrency 1
```

<sup>2. Amazon EC2 m6i.24xlarge was used for benchmarking purpose only. For multi-core instances, ipex optimizations automatically scale and leverage full instance resources.</sup>
