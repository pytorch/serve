
## GPT-Fast

[GPT fast](https://github.com/pytorch-labs/gpt-fast) is a simple and efficient pytorch-native transformer text generation.

It features:
* Very low latency
* <1000 lines of python
* No dependencies other than PyTorch and sentencepiece
* int8/int4 quantization
* Speculative decoding
* Supports multi-GPU inference through Tensor parallelism
* Supports Nvidia and AMD GPUs

More details about gpt-fast can be found in this [blog](https://pytorch.org/blog/accelerating-generative-ai-2/).
The examples has been tested on A10, A100 as well as H100.


#### Pre-requisites

`cd` to the example folder `examples/large_models/gpt_fast`

Install dependencies and upgrade torch to nightly build (currently required)
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
huggingface-cli login
./scripts/prepare.sh $MODEL_REPO
cd ..
```

### (Optional) Step 1.5: Quantize the model to int4

To speed up model loading and inference even further we can optionally quantize the model to int4 instead of int8. Please see the [blog post](https://pytorch.org/blog/accelerating-generative-ai-2/) for details on the potential accuracy loss.

```
cd gpt-fast
python quantize.py --checkpoint_path checkpoints/$MODEL_REPO/model.pth --mode int4
cd ..
```

The quantized model will show up as checkpoints/$MODEL_REPO/model_int4.pth. To enable it in the example you need to exchange the filename in the [`model_config.yaml`](./model_config.yaml) file.

### (Optional) Step 1.6: Speculative decoding

Another technique to speed-up inference that was implemented in gpt-fast is speculative decoding where a smaller draft model is used to provide token proposals which are only verified with a bigger main model.
Depending on draft and verifier model this can be faster due to the parallel nature of the check.
For more details on selecting the models check the [gpt-fast blog post](https://pytorch.org/blog/accelerating-generative-ai-2/).

In order to use speculative decoding we need a draft model which needs to be aligned with the verifier model. E.g. we can use `meta-llama/Llama-2-13b-chat-hf` and `meta-llama/Llama-2-7b-chat-hf` model.
For this example we prepare `meta-llama/Llama-2-13b-chat-hf` model as verifier and use the model generated in step 1 as draft model.

```
export MODEL_REPO=meta-llama/Llama-2-13b-chat-hf
cd gpt-fast
huggingface-cli login
./scripts/prepare.sh $MODEL_REPO
cd ..
```

### Step 2: Generate model archive
At this stage we're creating the model archive which includes the configuration of our model in [model_config.yaml](./model_config.yaml).
It's also the point where we need to decide if we want to deploy our model on a single or multiple GPUs.
For the single GPU case we can use the default configuration that can be found in [model_config.yaml](./model_config.yaml).
All configs enable the current prototyping feature FxGraphCache by setting fx_graph_cache to *true*.
This feature stores the TorchInductor output in a cache to speed up torch.compile times when rerunning the handler.

```
torch-model-archiver --model-name gpt_fast --version 1.0 --handler handler.py --config-file model_config.yaml --extra-files "gpt-fast/generate.py,gpt-fast/model.py,gpt-fast/quantize.py,gpt-fast/tp.py" --archive-format no-archive
mv gpt-fast/checkpoints gpt_fast/
```

If we want to use tensor parallel variant and split the model over multiple GPUs we need to set the grade of desired tensor parallelism in [model_config_tp.yaml](./model_config_tp.yaml) and use this configuration for creating the archive:
```
torch-model-archiver --model-name gpt_fast --version 1.0 --handler handler.py --config-file model_config_tp.yaml --extra-files "gpt-fast/generate.py,gpt-fast/model.py,gpt-fast/quantize.py,gpt-fast/tp.py" --archive-format no-archive
mv gpt-fast/checkpoints gpt_fast/
```

If we want to activate speculative decoding and have prepared the verifier model in step 1.6 we can go ahead and create the model archive using [model_config_speculative.yaml](./model_config_speculative.yaml) which combines tensor parallel and speculative decoding.
```
torch-model-archiver --model-name gpt_fast --version 1.0 --handler handler.py --config-file model_config_speculative.yaml --extra-files "gpt-fast/generate.py,gpt-fast/model.py,gpt-fast/quantize.py,gpt-fast/tp.py" --archive-format no-archive
mv gpt-fast/checkpoints gpt_fast/
```

### Step 3: Add the model archive to model store

```
mkdir model_store
mv gpt_fast model_store
```

### Step 4: Start torchserve

```
torchserve --start --ncs --model-store model_store --models gpt_fast
```

### Step 5: Run inference

```
curl "http://localhost:8080/predictions/gpt_fast" -T request.json
# Returns: The capital of France, Paris, is a city of romance, fashion, and art. The city is home to the Eiffel Tower, the Louvre, and the Arc de Triomphe. Paris is also known for its cafes, restaurants
```
