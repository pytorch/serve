# Example showing inference with vLLM

This folder contains multiple demonstrations showcasing the integration of [vLLM Engine](https://github.com/vllm-project/vllm) with TorchServe, running inference with continuous batching.
vLLM achieves high throughput using PagedAttention. More details can be found [here](https://vllm.ai/)

- demo1: [Mistral](mistral)
- demo2: [lora](lora)

### Supported vLLM Configuration
* LLMEngine configuration:
vLLM [EngineArgs](https://github.com/vllm-project/vllm/blob/258a2c58d08fc7a242556120877a89404861fbce/vllm/engine/arg_utils.py#L15) is defined in the section of `handler/vllm_engine_config` of model-config.yaml.


* Sampling parameters for text generation:
vLLM [SamplingParams](https://github.com/vllm-project/vllm/blob/258a2c58d08fc7a242556120877a89404861fbce/vllm/sampling_params.py#L27) is defined in the JSON format, for example, [prompt.json](lora/prompt.json).
