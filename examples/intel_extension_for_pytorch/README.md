# TorchServe with Intel® Extension for PyTorch*

TorchServe can be used with Intel® Extension for PyTorch* to give performance boost on Intel hardware.<sup>1</sup>
Here we show how to use TorchServe with Intel® Extension for PyTorch*.

<sup>1. While Intel® Extension for PyTorch* benefits all platforms, platforms with AVX512 benefit the most. </sup>

## Contents of this Document
* [Install Intel® Extension for PyTorch*](https://github.com/pytorch/serve/blob/master/examples/intel_extension_for_pytorch/README.md#install-intel-extension-for-pytorch)
* [Serving model with Intel® Extension for PyTorch*](https://github.com/pytorch/serve/blob/master/examples/intel_extension_for_pytorch/README.md#serving-model-with-intel-extension-for-pytorch)
* [TorchServe with Launcher](#torchserve-with-launcher)
* [TorchServe with Intel® Extension for PyTorch* and Intel GPUs](#torchserve-with-intel®-extension-for-pytorch-and-intel-gpus)
* [Performance Gain with Intel® Extension for PyTorch* and Intel GPU](https://github.com/pytorch/serve/blob/master/examples/intel_extension_for_pytorch/README.md#performance-gain-with-intel-extension-for-pytorch-and-intel-gpu)
* [Creating and Exporting INT8 model for Intel® Extension for PyTorch*](https://github.com/pytorch/serve/blob/master/examples/intel_extension_for_pytorch/README.md#creating-and-exporting-int8-model-for-intel-extension-for-pytorch)
* [Benchmarking with Launcher](#benchmarking-with-launcher)
* [Performance Boost with Intel® Extension for PyTorch* and Launcher](https://github.com/pytorch/serve/blob/master/examples/intel_extension_for_pytorch/README.md#performance-boost-with-intel-extension-for-pytorch-and-launcher)


## Install Intel® Extension for PyTorch*
Refer to the documentation [here](https://github.com/intel/intel-extension-for-pytorch/blob/master/docs/tutorials/installation.md).

## Serving model with Intel® Extension for PyTorch*
After installation, all it needs to use TorchServe with Intel® Extension for PyTorch* is to enable it in `config.properties`.
```
ipex_enable=true
```
Once Intel® Extension for PyTorch* is enabled, deploying PyTorch model follows the same procedure shown [here](https://pytorch.org/serve/use_cases.html). TorchServe with Intel® Extension for PyTorch* can deploy any model and do inference.

## TorchServe with Launcher
Launcher is a script to automate the process of tuning configuration setting on Intel hardware to boost performance. Tuning configurations such as OMP_NUM_THREADS, thread affinity, memory allocator can have a dramatic effect on performance. Refer to [Performance Tuning Guide](https://github.com/intel/intel-extension-for-pytorch/blob/master/docs/tutorials/performance_tuning/tuning_guide.md) and [Launch Script Usage Guide](https://github.com/intel/intel-extension-for-pytorch/blob/master/docs/tutorials/performance_tuning/launch_script.md) for details on performance tuning with launcher.

All it needs to use TorchServe with launcher is to set its configuration in `config.properties`.

Add the following lines in `config.properties` to use launcher with its default configuration.
```
ipex_enable=true
cpu_launcher_enable=true
```

Launcher by default uses `numactl` if it's installed to ensure socket is pinned and thus memory is allocated from local numa node. To use launcher without numactl, add the following lines in `config.properties`.
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

Below are some useful `cpu_launcher_args` to note. Italic values are default if applicable.
1. Memory Allocator: [ PTMalloc `--use_default_allocator` | *TCMalloc `--enable_tcmalloc`* | JeMalloc `--enable_jemalloc`]
   * PyTorch by default uses PTMalloc. TCMalloc/JeMalloc generally gives better performance.
2. OpenMP library: [GNU OpenMP `--disable_iomp` | *Intel OpenMP*]
   * PyTorch by default uses GNU OpenMP. Launcher by default uses Intel OpenMP. Intel OpenMP library generally gives better performance.
3. Node id: [`--node_id`]
   * Launcher by default uses all NUMA nodes. Limit memory access to local memories on the Nth Numa node to avoid Non-Uniform Memory Access (NUMA).

Refer to [Launch Script Usage Guide](https://github.com/intel/intel-extension-for-pytorch/blob/master/docs/tutorials/performance_tuning/launch_script.md) for a full list of tunable configuration of launcher. And refer to [Performance Tuning Guide](https://github.com/intel/intel-extension-for-pytorch/blob/master/docs/tutorials/performance_tuning/tuning_guide.md) for more details.

### Launcher Core Pinning to Boost Performance of TorchServe Multi Worker Inference 
When running [multi-worker inference](https://pytorch.org/serve/management_api.html#scale-workers) with Torchserve, launcher pin cores to workers to boost performance. Internally, launcher equally divides the number of cores by the number of workers such that each worker is pinned to assigned cores. Doing so avoids core overlap among workers which can significantly boost performance for TorchServe multi-worker inference. For example, assume running 4 workers on a machine with Intel(R) Xeon(R) Platinum 8180 CPU, 2 sockets, 28 cores per socket, 2 threads per core. Launcher will bind worker 0 to cores 0-13, worker 1 to cores 14-27, worker 2 to cores 28-41, and worker 3 to cores 42-55. 

CPU usage is shown below. 4 main worker threads were launched, each launching 14 threads affinitized to the assigned physical cores.
![26](https://user-images.githubusercontent.com/93151422/170373651-fd8a0363-febf-4528-bbae-e1ddef119358.gif)



#### Scaling workers
Additionally when dynamically [scaling the number of workers](https://pytorch.org/serve/management_api.html#scale-workers), cores that were pinned to killed workers by the launcher could be left unutilized. To address this problem, launcher internally restarts the workers to re-distribute cores that were pinned to killed workers to the remaining, alive workers. This is taken care internally, so users do not have to worry about this. 

Continuing with the above example with 4 workers, assume killing workers 2 and 3. If cores were not re-distributed after the scale down, cores 28-55 would be left unutilized. Instead, launcher re-distributes cores 28-55 to workers 0 and 1 such that now worker 0 binds to cores 0-27 and worker 1 binds to cores 28-55.<sup>2</sup> 

CPU usage is shown below. 4 main worker threads were initially launched. Then after scaling down the number of workers from 4 to 2, 2 main worker threads were launched, each launching 28 threads affinitized to the assigned physical cores.
![worker_scaling](https://user-images.githubusercontent.com/93151422/170374697-7497c2d5-4c17-421b-9993-1434d1f722f6.gif)

<sup>2. Serving is interrupted for few seconds while re-distributing cores to scaled workers.</sup>

Again, all it needs to use TorchServe with launcher core pinning for multiple workers as well as scaling workers is to set its configuration in `config.properties`.

Add the following lines in `config.properties` to use launcher with its default configuration. 
```
cpu_launcher_enable=true
```

## TorchServe with Intel® Extension for PyTorch* and Intel GPUs

TorchServe can also leverage Intel GPU for acceleration, providing additional performance benefits. To use TorchServe with Intel GPU, the machine must have the latest oneAPI Base Kit installed, activated, and ipex GPU installed.


### Installation and Setup for Intel GPU Support
**Install Intel oneAPI Base Kit:** 
Follow the installation instructions for your operating system from the [Intel oneAPI Base kit Installation](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.htm).

**Install the ipex GPU package to enable TorchServe to utilize Intel GPU for acceleration:** 
Follow the installation instructions for your operating system from the [ Intel® Extension for PyTorch* XPU/GPU Installation](https://intel.github.io/intel-extension-for-pytorch/index.html#installation?platform=gpu).

**Activate the Intel oneAPI Base Kit:** 
Activate the Intel oneAPI Base Kit using the following command:
   ```bash
   source /path/to/oneapi/setvars.sh
   ```

**Install xpu-smi:**
Install xpu-smi to let torchserve detect the number of Intel GPU devices present. xpu-smi provides information about the Intel GPU, including temperature, utilization, and other metrics.[xpu-smi Installation Guide](https://dgpu-docs.intel.com/driver/installation.html#ubuntu-package-repository)

**Enable Intel GPU Support in TorchServe:** 
To enable TorchServe to use Intel GPUs, set the following configuration in `config.properties`:
   ```
   ipex_enable=true
   ipex_gpu_enable=true
   ```
To enable metric reading, additionally set:
   ```
   system_metrics_cmd=${PATH to examples/intel_extension_for_pytorch/intel_gpu_metric_collector.py} --gpu ${Number of GPUs}
   ```

## Performance Gain with Intel® Extension for PyTorch* and Intel GPU

To understand the performance gain using Intel GPU, Torchserve recommended [apache benchmark](https://github.com/pytorch/serve/tree/master/benchmarks#benchmarking-with-apache-bench) is executed on FastRCNN FP32 model.

A `model_config.json` file is created, and the following parameters are added:

```
{
  "url": "https://torchserve.pytorch.org/mar_files/fastrcnn.mar",
  "requests": "10000",
  "concurrency": "100",
  "workers": "1",
  "batch_delay": "100",
  "batch_size": "1",
  "input": "../examples/image_classifier/kitten.jpg",
  "backend_profiling": "FALSE",
  "exec_env": "local"
}
```

Batch size can be changed according to the requirement.

Following lines are added to the `config.properties` to utilize IPEX and Intel GPU:

```
ipex_enable=true
ipex_gpu_enable=true
```

To reproduce the test, use the following command:

```
python benchmark-ab.py --config model_config.json --config_properties config.properties
```

This test is performed on a server containing Intel(R) Core (TM) i5-9600K CPU + Intel(R) Arc(TM) A770 Graphics and is compared with a Intel(R) Xeon(R) Gold 6438Y CPU server.
It is recommended to use only 1 worker per GPU, more than 1 worker per GPU is not validated and may cause performance degradation due to context switching.


| Model | Batch size | CPU Throughput(img/sec) | GPU Throughput(img/sec) | CPU TS Latency mean(ms) | GPU TS Latency mean(ms) | Throughput speedup ratio | Latency speedup ratio |
|:-----:|:----------:|:--------------:|:--------------:|:-------------------:|:-------------------:|:-------------------------:|:----------------------:|
| FastRCNN_FP32 | 1 | 15.74 | 2.89 | 6352.388 | 34636.68 | 5.45 | 5.45 |
|  | 2 | 17.69 | 2.67 | 5651.999 | 37520.781 | 6.63 | 6.64 |
|  | 4 | 18.57 | 2.39 | 5385.389 | 41886.902 | 7.77 | 7.78 |
|  | 8 | 18.68 | 2.32 | 5354.58 | 43146.797 | 8.05 | 8.06 |
|  | 16 | 19.26 | 2.39 | 5193.307 | 41903.752 | 8.06 | 8.07 |
|  | 32 | 19.06 | 2.49 | 5245.912 | 40172.39 | 7.65 | 7.66 |

<p align="center">
  <img src="https://github.com/pytorch/serve/assets/113945574/c30aeacc-9825-42b1-bde8-2d9dca09bb8a" />
</p>
Above graph shows the speedup ratio of throughput and latency while using Intel GPU. The speedup ratio is increasing steadily reaching almost 8x till batch size 8 and gives diminishing returns after. Further increasing the batch size to 64 results in `RuntimeError: Native API failed. Native API returns: -5 (PI_ERROR_OUT_OF_RESOURCES)` error as GPU is overloaded.

Note: The optimal configuration will vary depending on the hardware used.

## Creating and Exporting INT8 model for Intel® Extension for PyTorch*
Intel® Extension for PyTorch* supports both eager and torchscript mode. In this section, we show how to deploy INT8 model for Intel® Extension for PyTorch*. Refer to [here](https://github.com/intel/intel-extension-for-pytorch/blob/master/docs/tutorials/features/int8_overview.md) for more details on Intel® Extension for PyTorch* optimizations for quantization.

### 1. Creating a serialized file 
First create `.pt` serialized file using Intel® Extension for PyTorch* INT8 inference. Here we show two examples with BERT and ResNet50. 

#### BERT

```
import torch
import intel_extension_for_pytorch as ipex
from transformers import BertModel

# load the model
model = BertModel.from_pretrained('bert-base-uncased')
model = model.eval()

# define dummy input tensor to use for the model's forward call to record operations in the model for tracing
vocab_size = model.config.vocab_size
batch_size = 1
seq_length = 384
dummy_tensor = torch.randint(vocab_size, size=[batch_size, seq_length])

from intel_extension_for_pytorch.quantization import prepare, convert

# ipex supports two quantization schemes: static and dynamic
# default dynamic qconfig
qconfig = ipex.quantization.default_dynamic_qconfig

# prepare and calibrate
model = prepare(model, qconfig, example_inputs=dummy_tensor)

# convert and deploy
model = convert(model)

with torch.no_grad():
    model = torch.jit.trace(model, dummy_tensor, check_trace=False, strict=False)
    model = torch.jit.freeze(model)

torch.jit.save(model, 'bert_int8_jit.pt')
```

#### ResNet50

```
import torch
import intel_extension_for_pytorch as ipex
import torchvision.models as models

# load the model
model = models.resnet50(pretrained=True)
model = model.eval()

# define dummy input tensor to use for the model's forward call to record operations in the model for tracing
N, C, H, W = 1, 3, 224, 224
dummy_tensor = torch.randn(N, C, H, W)

from intel_extension_for_pytorch.quantization import prepare, convert

# ipex supports two quantization schemes: static and dynamic
# default static qconfig
qconfig = ipex.quantization.default_static_qconfig

# prepare and calibrate
model = prepare(model, qconfig, example_inputs=dummy_tensor, inplace=False)
 
n_iter = 100
for i in range(n_iter):
    model(dummy_tensor)
 
# convert and deploy
model = convert(model)

with torch.no_grad():
    model = torch.jit.trace(model, dummy_tensor)
    model = torch.jit.freeze(model)

torch.jit.save(model, 'rn50_int8_jit.pt')
```

### 2. Creating a Model Archive 
Once the serialized file ( `.pt`) is created, it can be used with `torch-model-archiver` as usual.

Use the following command to package `rn50_int8_jit.pt` into `rn50_ipex_int8.mar`.  
```
torch-model-archiver --model-name rn50_ipex_int8 --version 1.0 --serialized-file rn50_int8_jit.pt --handler image_classifier
```
Similarly, use the following command in the [Huggingface_Transformers directory](https://github.com/pytorch/serve/tree/master/examples/Huggingface_Transformers) to package `bert_int8_jit.pt` into `bert_ipex_int8.mar`.   

```
torch-model-archiver --model-name bert_ipex_int8 --version 1.0 --serialized-file bert_int8_jit.pt --handler ./Transformer_handler_generalized.py --extra-files "./setup_config.json,./Seq_classification_artifacts/index_to_name.json"
```

### 3. Start TorchServe to serve the model 
Make sure to set `ipex_enable=true` in `config.properties`. Use the following command to start TorchServe with Intel® Extension for PyTorch*. 
```
torchserve --start --ncs --model-store model_store --ts-config config.properties
```

### 4. Registering and Deploying model
Registering and deploying the model follows the same steps shown [here](https://pytorch.org/serve/use_cases.html).

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

### Benchmarking with Launcher Core Pinning
As described previously in [TorchServe with Launcher](#torchserve-with-launcher), launcher core pinning boosts performance of multi-worker inference. We'll demonstrate launcher core pinning with TorchServe benchmark, but keep in mind that launcher core pinning is a generic feature applicable to any TorchServe multi-worker inference use case.

For example, assume running 4 workers 
```
python benchmark-ab.py --workers 4
```
on a machine with Intel(R) Xeon(R) Platinum 8180 CPU, 2 sockets, 28 cores per socket, 2 threads per core. Launcher will bind worker 0 to cores 0-13, worker 1 to cores 14-27, worker 2 to cores 28-41, and worker 3 to cores 42-55. 

All it needs to use TorchServe with launcher's core pinning is to enable launcher in `config.properties`.

Add the following lines to `config.properties` in the benchmark directory to use launcher's core pinning:
```
cpu_launcher_enable=true
```

CPU usage is shown as below:
![launcher_core_pinning](https://user-images.githubusercontent.com/93151422/159063975-e7e8d4b0-e083-4733-bdb6-4d92bdc10556.gif)

4 main worker threads were launched, then each launched a num_physical_cores/num_workers number (14) of threads affinitized to the assigned physical cores. 

<pre><code>
$ cat logs/model_log.log
2022-03-24 10:41:32,223 - __main__ - INFO - Use TCMalloc memory allocator
2022-03-24 10:41:32,223 - __main__ - INFO - OMP_NUM_THREADS=14
2022-03-24 10:41:32,223 - __main__ - INFO - Using Intel OpenMP
2022-03-24 10:41:32,223 - __main__ - INFO - KMP_AFFINITY=granularity=fine,compact,1,0
2022-03-24 10:41:32,223 - __main__ - INFO - KMP_BLOCKTIME=1
2022-03-24 10:41:32,223 - __main__ - INFO - LD_PRELOAD=<VIRTUAL_ENV>/lib/libiomp5.so:<VIRTUAL_ENV>/lib/libtcmalloc.so
2022-03-24 10:41:32,223 - __main__ - INFO - <b>numactl -C 0-13 -m 0</b> <VIRTUAL_ENV>/bin/python -u <VIRTUAL_ENV>/lib/python/site-packages/ts/model_service_worker.py --sock-type unix --sock-name /tmp/.ts.sock.9000

2022-03-24 10:49:03,760 - __main__ - INFO - Use TCMalloc memory allocator
2022-03-24 10:49:03,761 - __main__ - INFO - OMP_NUM_THREADS=14
2022-03-24 10:49:03,762 - __main__ - INFO - Using Intel OpenMP
2022-03-24 10:49:03,762 - __main__ - INFO - KMP_AFFINITY=granularity=fine,compact,1,0
2022-03-24 10:49:03,762 - __main__ - INFO - KMP_BLOCKTIME=1
2022-03-24 10:49:03,762 - __main__ - INFO - LD_PRELOAD=<VIRTUAL_ENV>/lib/libiomp5.so:<VIRTUAL_ENV>/lib/libtcmalloc.so
2022-03-24 10:49:03,763 - __main__ - INFO - <b>numactl -C 14-27 -m 0</b> <VIRTUAL_ENV>/bin/python -u <VIRTUAL_ENV>/lib/python/site-packages/ts/model_service_worker.py --sock-type unix --sock-name /tmp/.ts.sock.9001

2022-03-24 10:49:26,274 - __main__ - INFO - Use TCMalloc memory allocator
2022-03-24 10:49:26,274 - __main__ - INFO - OMP_NUM_THREADS=14
2022-03-24 10:49:26,274 - __main__ - INFO - Using Intel OpenMP
2022-03-24 10:49:26,274 - __main__ - INFO - KMP_AFFINITY=granularity=fine,compact,1,0
2022-03-24 10:49:26,274 - __main__ - INFO - KMP_BLOCKTIME=1
2022-03-24 10:49:26,274 - __main__ - INFO - LD_PRELOAD=<VIRTUAL_ENV>/lib/libiomp5.so:<VIRTUAL_ENV>/lib/libtcmalloc.so
2022-03-24 10:49:26,274 - __main__ - INFO - <b>numactl -C 28-41 -m 1</b> <VIRTUAL_ENV>/bin/python -u <VIRTUAL_ENV>/lib/python/site-packages/ts/model_service_worker.py --sock-type unix --sock-name /tmp/.ts.sock.9002

2022-03-24 10:49:42,975 - __main__ - INFO - Use TCMalloc memory allocator
2022-03-24 10:49:42,975 - __main__ - INFO - OMP_NUM_THREADS=14
2022-03-24 10:49:42,975 - __main__ - INFO - Using Intel OpenMP
2022-03-24 10:49:42,975 - __main__ - INFO - KMP_AFFINITY=granularity=fine,compact,1,0
2022-03-24 10:49:42,975 - __main__ - INFO - KMP_BLOCKTIME=1
2022-03-24 10:49:42,975 - __main__ - INFO - LD_PRELOAD=<VIRTUAL_ENV>/lib/libiomp5.so:<VIRTUAL_ENV>/lib/libtcmalloc.so
2022-03-24 10:49:42,975 - __main__ - INFO - <b>numactl -C 42-55 -m 1</b> <VIRTUAL_ENV>/bin/python -u <VIRTUAL_ENV>/lib/python/site-packages/ts/model_service_worker.py --sock-type unix --sock-name /tmp/.ts.sock.9003
</code></pre>

## Performance Boost with Intel® Extension for PyTorch* and Launcher

![pdt_perf](https://user-images.githubusercontent.com/93151422/159067306-dfd604e3-8c66-4365-91ae-c99f68d972d5.png)


Above shows performance improvement of Torchserve with Intel® Extension for PyTorch* and launcher on ResNet50 and BERT-base-uncased. Torchserve official [apache-bench benchmark](https://github.com/pytorch/serve/tree/master/benchmarks#benchmarking-with-apache-bench) on Amazon EC2 m6i.24xlarge was used to collect the results<sup>2</sup>. Add the following lines in ```config.properties``` to reproduce the results. Notice that launcher is configured such that a single instance uses all physical cores on a single socket to avoid cross socket communication and core overlap. 

```
ipex_enable=true
cpu_launcher_enable=true
cpu_launcher_args=--node_id 0 --enable_jemalloc
```
Use the following command to reproduce the results.
```
python benchmark-ab.py --url {modelUrl} --input {inputPath} --concurrency 1
```

For example, run the following command to reproduce latency performance of ResNet50 with data type of Intel® Extension for PyTorch* int8 and batch size of 1. Refer to [Creating and Exporting INT8 model for Intel® Extension for PyTorch*](https://github.com/pytorch/serve/blob/master/examples/intel_extension_for_pytorch/README.md#creating-and-exporting-int8-model-for-intel-extension-for-pytorch) for steps to creating ```rn50_ipex_int8.mar``` file for ResNet50 with Intel® Extension for PyTorch* int8 data type.
```
python benchmark-ab.py --url 'file:///model_store/rn50_ipex_int8.mar' --concurrency 1
```

For example, run the following command to reproduce latency performance of BERT with data type of Intel® Extension for PyTorch* int8 and batch size of 1. Refer to [Creating and Exporting INT8 model for Intel® Extension for PyTorch*](https://github.com/pytorch/serve/blob/master/examples/intel_extension_for_pytorch/README.md#creating-and-exporting-int8-model-for-intel-extension-for-pytorch) for steps to creating ```bert_ipex_int8.mar``` file for BERT with Intel® Extension for PyTorch* int8 data type.
```
python benchmark-ab.py --url 'file:///model_store/bert_ipex_int8.mar' --input '../examples/Huggingface_Transformers/Seq_classification_artifacts/sample_text_captum_input.txt' --concurrency 1
```

<sup>3. Amazon EC2 m6i.24xlarge was used for benchmarking purpose only. For multi-core instances, Intel® Extension for PyTorch* optimizations automatically scale and leverage full instance resources.</sup>
