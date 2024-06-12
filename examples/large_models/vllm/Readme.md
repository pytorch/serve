# Example showing inference with vLLM

This folder contains multiple demonstrations showcasing the integration of [vLLM Engine](https://github.com/vllm-project/vllm) with TorchServe, running inference with continuous batching.
vLLM achieves high throughput using PagedAttention. More details can be found [here](https://vllm.ai/).
The vLLM integration uses our new asynchronous worker communication mode which decoupled communication between frontend and backend from running the actual inference.
By using this new feature TorchServe is capable to feed incoming requests into the vLLM engine while asynchronously running the engine in the backend.
As long as a single request is inside the engine it will continue to run and asynchronously stream out the results until the request is finished.
New requests are added to the engine in a continuous fashion similar to the continuous batching mode shown in other examples.
For all examples distributed inference can be enabled by following the instruction [here](./Readme.md#distributed-inference)

- demo1: [Meta-Llama3](llama3)
- demo2: [Mistral](mistral)
- demo3: [lora](lora)

### Supported vLLM Configuration
* LLMEngine configuration:
vLLM [EngineArgs](https://github.com/vllm-project/vllm/blob/258a2c58d08fc7a242556120877a89404861fbce/vllm/engine/arg_utils.py#L15) is defined in the section of `handler/vllm_engine_config` of model-config.yaml.


* Sampling parameters for text generation:
vLLM [SamplingParams](https://github.com/vllm-project/vllm/blob/258a2c58d08fc7a242556120877a89404861fbce/vllm/sampling_params.py#L27) is defined in the JSON format, for example, [prompt.json](lora/prompt.json).

### Distributed Inference
All examples can be easily distributed over multiple GPUs by enabling tensor parallelism in vLLM.
To enable distributed inference the following additions need to made to the model-config.yaml of the examples where 4 is the number of desired GPUs to use for the inference:

```yaml
# TorchServe frontend parameters
...
parallelType: "custom"
parallelLevel: 4

handler:
    ...
    vllm_engine_config:
        ...
        tensor_parallel_size: 4
```

### Multi-worker Note:
While this example in theory works with multiple workers it would distribute the incoming requests in a round robin fashion which might lead to non optimal worker/hardware utilization.
It is therefore advised to only use a single worker per engine and utilize tensor parallelism to distribute the model over multiple GPUs as described in the previous section.
This will result in better hardware utilization and inference performance.
