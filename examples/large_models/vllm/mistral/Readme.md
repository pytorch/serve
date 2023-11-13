# Example showing inference with vLLM with mistralai/Mistral-7B-v0.1 model

This is an example showing how to integrate [vLLM](https://github.com/vllm-project/vllm) with TorchServe and run inference on `mistralai/Mistral-7B-v0.1` model.
vLLM achieves high throughput using PagedAttention. More details can be found [here](https://vllm.ai/)

Install vLLM with the following

```
pip install -r requirements.txt
```
### Step 1: Login to HuggingFace

Login with a HuggingFace account
```
huggingface-cli login
# or using an environment variable
huggingface-cli login --token $HUGGINGFACE_TOKEN
```

```bash
python ../../Huggingface_accelerate/Download_model.py --model_path model --model_name mistralai/Mistral-7B-v0.1
```
Model will be saved in the following path, `mistralai/Mistral-7B-v0.1`.

### Step 2: Generate MAR file

Add the downloaded path to " model_path:" in `model-config.yaml` and run the following.

```bash
torch-model-archiver --model-name mistral7b --version 1.0 --handler custom_handler.py --config-file model-config.yaml -r requirements.txt --archive-format tgz
```

### Step 3: Add the mar file to model store

```bash
mkdir model_store
mv mistral7b.tar.gz model_store
```

### Step 3: Start torchserve

Update config.properties and start torchserve

```bash
torchserve --start --ncs --ts-config config.properties --model-store model_store --models mistral7b.tar.gz
```

### Step 4: Run inference

```bash
curl -v "http://localhost:8080/predictions/mistral7b" -T sample_text.txt
```

results in the following output
```
Mayonnaise is made of eggs, oil, vinegar, salt and pepper. Using an electric blender, combine all the ingredients and beat at high speed for 4 to 5 minutes.

Try it with some mustard and paprika mixed in, and a bit of sweetener if you like. But use real mayonnaise or it isn’t the same. Marlou

What in the world is mayonnaise?
```
