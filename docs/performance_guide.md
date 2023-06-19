# [Performance Guide](#performance-guide)
In case you're interested in optimizing the memory usage, latency or throughput of a PyTorch model served with TorchServe, this is the guide for you.

## Optimizing PyTorch

There are many tricks to optimize PyTorch models for production including but not limited to distillation, quantization, fusion, pruning, setting environment variables and we encourage you to benchmark and see what works best for you.

In general it's hard to optimize models and the easiest approach can be exporting to some runtime like ORT, TensorRT, IPEX or FasterTransformer. We have many examples for how to integrate these runtimes on the [TorchServe github page](https://github.com/pytorch/serve/tree/master/examples). If your favorite runtime is not supported please feel free to open a PR.

<h4>ONNX and ORT support</h4>

TorchServe has native support for ONNX models which can be loaded via ORT for both accelerated CPU and GPU inference. ONNX operates a bit differently from a regular PyTorch model in that when you're running the conversion you need to explicitly set and name your input and output dimensions. See [this example](https://github.com/pytorch/serve/blob/master/test/pytest/test_onnx.py).

At a high level what TorchServe allows you to do is
1. Package serialized ONNX weights `torch-model-archiver --serialized-file model.onnx ...`
2. Load those weights from `base_handler.py` using `ort_session = ort.InferenceSession(self.model_pt_path, providers=providers, sess_options=sess_options)` which supports reasonable defaults for both CPU and GPU inference
3. Allow you define custom pre and post processing functions to pass in data in the format your onnx model expects with a custom handler

 <h4>TensorRT<h4>

TorchServe also supports models optimized via TensorRT. To leverage the TensorRT runtime you can convert your model by [following these instructions](https://github.com/pytorch/TensorRT) and once you're done you'll have serialized weights which you can load with [`torch.jit.load()`](https://pytorch.org/TensorRT/getting_started/getting_started_with_python_api.html#getting-started-with-python-api).

After a conversion there is no difference in how PyTorch treats a Torchscript model vs a TensorRT model.

 <h4>Better Transformer<h4>

Better Transformer from PyTorch implements a backwards-compatible fast path of `torch.nn.TransformerEncoder` for Transformer Encoder Inference and does not require model authors to modify their models. BetterTransformer improvements can exceed 2x in speedup and throughput for many common execution scenarios.
You can find more information on Better Transformer [here](https://pytorch.org/blog/a-better-transformer-for-fast-transformer-encoder-inference/) and [here](https://github.com/pytorch/serve/tree/master/examples/Huggingface_Transformers#speed-up-inference-with-better-transformer).

## Optimizing TorchServe

The main settings you should vary if you're trying to improve the performance of TorchServe from the `config.properties` are the `batch_size` and `batch_delay`. A larger batch size means a higher throughput at the cost of lower latency.

The second most important settings are  number of workers and number of gpus which will have a dramatic impact on CPU and GPU performance.

<h4>Concurrency And Number of Workers</h4>

TorchServe exposes configurations that allow the user to configure the number of worker threads on CPU and GPUs. There is an important config property that can speed up the server depending on the workload.
*Note: the following property has bigger impact under heavy workloads.*

<h4>TorchServe On CPU </h4>

If working with TorchServe on a CPU here are some things to consider that could improve performance:
* In a hyperthreading enabled system, avoid logical cores by setting thread affinity to physical cores only via core pinning.
* In a multi-socket system with NUMA, avoid cross-socket remote memory access by setting thread affinity to a specific socket via core pinning.

These principles can be automatically configured via an easy to use launch script which has already been integrated into TorchServe. For more information take a look at this [case study](https://pytorch.org/tutorials/intermediate/torchserve_with_ipex#grokking-pytorch-intel-cpu-performance-from-first-principles) which dives into these points further with examples and explanations from first principles.

<h4>TorchServe on GPU</h4>

There is a config property called `number_of_gpu` that tells the server to use a specific number of GPUs per model. In cases where we register multiple models with the server, this will apply to all the models registered. If this is set to a low value (ex: 0 or 1), it will result in under-utilization of GPUs. On the contrary, setting to a high value (>= max GPUs available on the system) results in as many workers getting spawned per model. Clearly, this will result in unnecessary contention for GPUs and can result in sub-optimal scheduling of threads to GPU.
```
ValueToSet = (Number of Hardware GPUs) / (Number of Unique Models)
```

<h6> NVIDIA MPS</h6>

While NVIDIA GPUs allow multiple processes to run on CUDA kernels, this comes with its own drawbacks namely:
* The execution of the kernels is generally serialized
* Each processes creates its own CUDA context which occupies additional GPU memory

To get around these drawbacks, you can utilize the NVIDIA Multi-Process Service (MPS) to increase performance. You can find more information on how to utilize NVIDIA MPS with TorchServe  [here](mps.md).

<h6> NVIDIA DALI</h6>

The NVIDIA Data Loading Library (DALI) is a library for data loading and pre-processing to accelerate deep learning applications. It can be used as a portable drop-in replacement for built in data loaders and data iterators in popular deep learning frameworks. DALI provides a collection of highly optimized building blocks for loading and processing image, video and audio data.
You can find an example of DALI optimization integration with TorchServe [here](https://github.com/pytorch/serve/tree/master/examples/nvidia_dali).


## Benchmarking

To make comparing various model and TorchServe configurations easier to compare, we've added a few helper scripts that output performance data like p50, p90, p99 latency in a clean report [here](https://github.com/pytorch/serve/tree/master/benchmarks) and mostly require you to determine some configuration either via JSON or YAML.
You can find more information on TorchServe benchmarking [here](https://github.com/pytorch/serve/blob/master/benchmarks/README.md#torchserve-model-server-benchmarking).


## Profiling

TorchServe has native support for the PyTorch profiler which will help you find performance bottlenecks in your code.

```
export ENABLE_TORCH_PROFILER=TRUE
```

Visit this [link]( https://github.com/pytorch/kineto/tree/main/tb_plugin) to learn more about the PyTorch profiler.

## More Resources

<h4>TorchServe on the Animated Drawings App</h4>

For some insight into fine tuning TorchServe performance in an application, take a look at this [article](https://pytorch.org/blog/torchserve-performance-tuning/). The case study shown here uses the Animated Drawings App form Meta to improve TorchServe Performance.
