
## GPT-Fast

[GPT fast](https://github.com/pytorch-labs/gpt-fast) is a simple and efficient pytorch-native transformer text generation.

It features:
* Very low latency
* <1000 lines of python
* No dependencies other than PyTorch and sentencepiece
* int8/int4 quantization
* Speculative decoding
* Tensor parallelism
* Supports Nvidia and AMD GPUs

More details about gpt-fast can be found in this [blog](https://pytorch.org/blog/accelerating-generative-ai-2/)


#### Pre-requisites

`cd` to the example folder `examples/large_models/gpt_fast`

Install dependencies and upgrade torch to nightlies (currently required)
```
git clone https://github.com/pytorch-labs/gpt-fast/
pip install sentencepiece huggingface_hub
pip uninstall torchtext torchdata torch torchvision torchaudio -y
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121 --ignore-installed
```

### Step 1: Download  and convert the weights

Currently supported models:
```
openlm-research/open_llama_7b
meta-llama/Llama-2-7b-chat-hf
meta-llama/Llama-2-13b-chat-hf
meta-llama/Llama-2-70b-chat-hf
codellama/CodeLlama-7b-Python-hf
codellama/CodeLlama-34b-Python-hf
```
Prepare weights:
```
export MODEL_REPO=meta-llama/Llama-2-7b-chat-hf
cd gpt-fast
./scripts/prepare.sh $MODEL_REPO
cd ..
```

### (Optional) Step 1.5: Quantize the model

To speed up loading the model as well as running inference with it we can optionally quantize the model to int8 datatype with very little to no accuracy degradation.

```
cd gpt-fast
python quantize.py --checkpoint_path checkpoints/$MODEL_REPO/model.pth --mode int8
cd ..
```

The quantized model will show up as checkpoints/$MODEL_REPO/model_int8.pth. To enable it in the example you need to exchange the filename in the [`model_config.yaml`](./model_config.yaml) file.


### Step 2: Generate model archive

```
torch-model-archiver --model-name gpt-fast --version 1.0 --handler handler.py --config-file model_config.yaml --extra-files "gpt-fast/generate.py,gpt-fast/model.py,gpt-fast/quantize.py,gpt-fast/tp.py" --archive-format tgz
```

### Step 3: Add the model archive to model store

```
mkdir model_store
mv gpt-fast.tar.gz model_store
```

### Step 4: Start torchserve

```
torchserve --start --ncs --model-store model_store --models gpt-fast.tar.gz
```

### Step 5: Run inference

```
curl "http://localhost:8080/predictions/gpt-fast" -T request.json
# Returns: The capital of France, Paris, is a city of romance, fashion, and art. The city is home to the Eiffel Tower, the Louvre, and the Arc de Triomphe. Paris is also known for its cafes, restaurants
```
