# Example showing inference with vLLM on Mistral model

This is an example showing how to integrate [vLLM](https://github.com/vllm-project/vllm) with TorchServe and run inference on model `mistralai/Mistral-7B-v0.1` with continuous batching.
This examples supports distributed inference by following [this instruction](../Readme.md#distributed-inference)

### Step 1: Download Model from HuggingFace

Login with a HuggingFace account
```
huggingface-cli login
# or using an environment variable
huggingface-cli login --token $HUGGINGFACE_TOKEN
```

```bash
python ../../utils/Download_model.py --model_path model --model_name mistralai/Mistral-7B-v0.1 --use_auth_token True
```

### Step 2: Generate model artifacts

Add the downloaded path to "model_path:" in `model-config.yaml` and run the following.

```bash
torch-model-archiver --model-name mistral --version 1.0 --handler vllm_handler --config-file model-config.yaml --archive-format no-archive
mv model mistral
```

### Step 3: Add the model artifacts to model store

```bash
mkdir model_store
mv mistral model_store
```

### Step 4: Start torchserve

```bash
torchserve --start --ncs --ts-config ../config.properties --model-store model_store --models mistral
```

### Step 5: Run inference

```bash
python ../../utils/test_llm_streaming_response.py -m mistral -o 50 -t 2 -n 4 --prompt-text "@prompt.json" --prompt-json
```
