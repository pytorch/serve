# Performance Guide
In case you're interested in optimizing the memory usage, latency or throughput of a PyTorch model served with TorchServe, this is the guide for you.
## Optimizing PyTorch
There are many tricks to optimize PyTorch models for production including but not limited to distillation, quantization, fusion, pruning, setting environment variables and we encourage you to benchmark and see what works best for you. An experimental tool that may make this process is easier is https://github.com/pytorch/serve/tree/master/experimental/torchprep

In general it's hard to optimize models and the easiest approach can be exporting to some runtime like ORT, TensorRT, IPEX, FasterTransformer and we have many examples for how to integrate these runtimes in https://github.com/pytorch/serve/tree/master/examples. If your favorite runtime is not supported please feel free to open a PR.

## Optimizing TorchServe
The main settings you should vary if you're trying to improve the performance of TorchServe from the `config.properties` are the `batch_size` and `batch_delay`. A larger batch size means a higher throughput at the cost of lower latency.

The second most important settings are  number of workers and number of gpus which will have a dramatic impact on CPU and GPU performance. To configure them:

### Concurrency And Number of Workers
TorchServe exposes configurations that allow the user to configure the number of worker threads on CPU and GPUs. There is an important config property that can speed up the server depending on the workload.
*Note: the following property has bigger impact under heavy workloads.*

**CPU**: there is a config property called `workers` which sets the number of worker threads for a model. The best value to set `workers` to is to start with `num physical cores / 2` and increase it as much possible after setting `torch.set_num_threads(1)` in your handler.

**GPU**: there is a config property called `number_of_gpu` that tells the server to use a specific number of GPUs per model. In cases where we register multiple models with the server, this will apply to all the models registered. If this is set to a low value (ex: 0 or 1), it will result in under-utilization of GPUs. On the contrary, setting to a high value (>= max GPUs available on the system) results in as many workers getting spawned per model. Clearly, this will result in unnecessary contention for GPUs and can result in sub-optimal scheduling of threads to GPU.
```
ValueToSet = (Number of Hardware GPUs) / (Number of Unique Models)
```

## Benchmarking
To make comparing various model and TorchServe configurations easier to compare, we've added a few helper scripts that output performance data like p50, p90, p99 latency in a clean report here and mostly require you to determine some configuration either via JSON or YAML https://github.com/pytorch/serve/tree/master/benchmarks.

If you'd like to run performance benchmarks locally https://github.com/pytorch/serve/tree/master/benchmarks

If you'd like to run performance benchmarks using an automated suite on AWS https://github.com/pytorch/serve/tree/master/benchmarks/automated

For the most part the core TorchServe team uses local benchmarks for experiments and the automated benchmarks for official releases.

## Profiling
TorchServe has native support for the PyTorch profiler which will help you find performance bottlenecks in your code.

```
export ENABLE_TORCH_PROFILER=TRUE
```

To learn more about the PyTorch profiler https://github.com/pytorch/kineto/tree/main/tb_plugin