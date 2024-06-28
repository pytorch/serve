# LLM Deployment with TorchServe

This document describes how to easily serve large language models (LLM) like Meta-Llama3 with TorchServe.
Besides a quick start guide using our VLLM integration we also provide a list of examples which describe other methods to deploy LLMs with TorchServe.

## Quickstart LLM Deployment

TorchServe offers easy LLM deployment through its VLLM integration.
Through the integration of our [LLM launcher script](https://github.com/pytorch/serve/blob/7a9b145204b4d7cfbb114fe737cf980221e6181e/ts/llm_launcher.py) users are able to deploy any model supported by VLLM with a single command.
The launcher can either be used standalone or in combination with our provided TorchServe GPU docker image.

To launch the docker we first need to build it:
```bash
docker build . -f docker/Dockerfile.llm -t ts/llm
```

Models are usually loaded from the HuggingFace hub and are cached in a [docker volume](https://docs.docker.com/storage/volumes/) for faster reload.
If you want to access gated models like the Meta-Llama3 model you need to provide a HuggingFace hub token:
```bash
export token=<HUGGINGFACE_HUB_TOKEN>
```

You can then go ahead and launch a TorchServe instance serving your selected model:
```bash
docker run --rm -ti --gpus all -e HUGGING_FACE_HUB_TOKEN=$token -p 8080:8080 -v data:/data ts/llm --model_id meta-llama/Meta-Llama-3-8B-Instruct --disable_token
```

To change the model you just need to exchange the identifier given to the `--model_id` parameter.
You can test the model with:
```bash
curl -X POST -d '{"prompt":"Hello, my name is", "max_new_tokens": 50}' --header "Content-Type: application/json" "http://localhost:8080/predictions/model"
```

You can change any of the sampling argument for the request by using the [VLLM SamplingParams keywords](https://docs.vllm.ai/en/stable/dev/sampling_params.html#vllm.SamplingParams).
E.g. for setting the sampling temperature to 0 we can do:
```bash
curl -X POST -d '{"prompt":"Hello, my name is", "max_new_tokens": 50, "temperature": 0}' --header "Content-Type: application/json" "http://localhost:8080/predictions/model"
```

TorchServe's LLM launcher scripts offers some customization options as well.
To rename the model endpoint from `predictions/model` to something else you can add `--model_name <SOME_NAME>` to the `docker run` command.

The launcher script can also be used outside a docker container by calling this after installing TorchServe following the [installation instruction](https://github.com/pytorch/serve/blob/feature/single_cmd_llm_deployment/README.md#-quick-start-with-torchserve).
```bash
python -m ts.llm_launcher --disable_token
```

Please note that the launcher script as well as the docker command will automatically run on all available GPUs so make sure to restrict the visible number of device by setting CUDA_VISIBLE_DEVICES.

For further customization of the handler and adding 3rd party dependencies you can have a look at out [VLLM example](https://github.com/pytorch/serve/tree/master/examples/large_models/vllm).

## Supported models
The quickstart launcher should allow to launch any model which is [supported by VLLM](https://docs.vllm.ai/en/latest/models/supported_models.html).
Here is a list of model identifiers tested by the TorchServe team:

* meta-llama/Meta-Llama-3-8B
* meta-llama/Meta-Llama-3-8B-Instruct
* meta-llama/Llama-2-7b-hf
* meta-llama/Llama-2-7b-chat-hf
* mistralai/Mistral-7B-v0.1
* mistralai/Mistral-7B-Instruct-v0.1

## Other ways to deploy LLMs with TorchServe

TorchServe offers a variety of example on how to deploy large models.
Here is a list of the current examples:

* [Llama 2/3 chat bot](https://github.com/pytorch/serve/tree/master/examples/LLM/llama)
* [GPT-fast](https://github.com/pytorch/serve/tree/master/examples/large_models/gpt_fast)
* [Inferentia2](https://github.com/pytorch/serve/tree/master/examples/large_models/inferentia2)
* [IPEX optimized](https://github.com/pytorch/serve/tree/master/examples/large_models/ipex_llm_int8)
* [Tensor Parallel Llama](https://github.com/pytorch/serve/tree/master/examples/large_models/tp_llama)
* [VLLM Integration](https://github.com/pytorch/serve/tree/master/examples/large_models/vllm)
