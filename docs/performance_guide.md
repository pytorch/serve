# Performance Guide
Lorem ipsum

## Optimizing PyTorch

## Optimizing TorchServe
### Concurrency And Number of Workers
TorchServe exposes configurations that allow the user to configure the number of worker threads on CPU and GPUs. There is an important config property that can speed up the server depending on the workload.
*Note: the following property has bigger impact under heavy workloads.*

**CPU**: there is a config property called `workers` which sets the number of worker threads for a model. The best value to set `workers` to is to start with `num physical cores / 2` and increase it as much possible after setting `torch.set_num_threads(1)` in your handler.

**GPU**: there is a config property called `number_of_gpu` that tells the server to use a specific number of GPUs per model. In cases where we register multiple models with the server, this will apply to all the models registered. If this is set to a low value (ex: 0 or 1), it will result in under-utilization of GPUs. On the contrary, setting to a high value (>= max GPUs available on the system) results in as many workers getting spawned per model. Clearly, this will result in unnecessary contention for GPUs and can result in sub-optimal scheduling of threads to GPU.
```
ValueToSet = (Number of Hardware GPUs) / (Number of Unique Models)
```

## Profiling