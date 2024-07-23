# Example showing inference with vLLM on LoRA model

This is an example showing how to integrate [vLLM](https://github.com/vllm-project/vllm) with TorchServe and run inference on model `Llama-2-7b-hf` + LoRA model `llama-2-7b-sql-lora-test` with continuous batching.
This examples supports distributed inference by following [this instruction](../Readme.md#distributed-inference)

### Step 0: Install vLLM

To leverage the power of vLLM we fist need to install it using pip in out development environment
```bash
python -m pip install -r ../requirements.txt
```
For later deployments we can make vLLM part of the deployment environment by adding the requirements.txt while building the model archive in step 2 (see [here](../../../../model-archiver/README.md#model-specific-custom-python-requirements) for details) or we can make it part of a docker image like [here](../../../../docker/Dockerfile.llm).

### Step 1: Download Model from HuggingFace

Login with a HuggingFace account
```
huggingface-cli login
# or using an environment variable
huggingface-cli login --token $HUGGINGFACE_TOKEN
```

```bash
python ../../utils/Download_model.py --model_path model --model_name meta-llama/Llama-2-7b-chat-hf --use_auth_token True
mkdir adapters && cd adapters
python ../../../utils/Download_model.py --model_path model --model_name yard1/llama-2-7b-sql-lora-test --use_auth_token True
cd ..
```

### Step 2: Generate model artifacts

Add the downloaded path to "model_path:" and "adapter_1:" in `model-config.yaml` and run the following.

```bash
torch-model-archiver --model-name llama-7b-lora --version 1.0 --handler vllm_handler --config-file model-config.yaml --archive-format no-archive
mv model llama-7b-lora
mv adapters llama-7b-lora
```

### Step 3: Add the model artifacts to model store

```bash
mkdir model_store
mv llama-7b-lora model_store
```

### Step 4: Start torchserve

```bash
torchserve --start --ncs --ts-config ../config.properties --model-store model_store --models llama-7b-lora --disable-token-auth --enable-model-api
```

### Step 5: Run inference

```bash
python ../../utils/test_llm_streaming_response.py -m lora -o 50 -t 2 -n 4 --prompt-text "@prompt.json" --prompt-json
```
