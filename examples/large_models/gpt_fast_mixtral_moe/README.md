
## Mixtral-MOE

We will be using [Mixtral-MOE](https://huggingface.co/docs/transformers/en/model_doc/mixtral).

It features:
* 8 experts per MLP
* 45 billion parameters
* compute required is the same as that of a 14 billion parameter model
* Sliding Window Attention
* GQA
* Byte-fallback BPE tokenizer

As a low-level framework we will be using [GPT fast](https://github.com/pytorch-labs/gpt-fast).



#### Pre-requisites

- PyTorch 2.3
- CUDA >= 11.8

`cd` to the example folder `examples/large_models/gpt_fast_mixtral_moe`

Install dependencies
```
git clone https://github.com/pytorch-labs/gpt-fast/
pip install sentencepiece huggingface_hub
```

### Step 1: Download  and convert the weights

Currently supported models:
```
mistralai/Mixtral-8x7B-v0.1
```
Prepare weights:
```
export MODEL_REPO=mistralai/Mixtral-8x7B-v0.1
huggingface-cli login
python gpt-fast/mixtral-moe/scripts/download.py --repo_id $MODEL_REPO
python gpt-fast/mixtral-moe/scripts/convert_hf_checkpoint.py --checkpoint_dir checkpoints/$MODEL_REPO
```

### Step 1.5: Quantize the model to int8

To speed up model loading and inference even further we can optionally quantize the model to int8. Please see the [blog post](https://pytorch.org/blog/accelerating-generative-ai-2/) for details on the potential accuracy loss.

```
python gpt-fast/mixtral-moe/quantize.py --checkpoint_path checkpoints/$MODEL_REPO/model.pth --mode int8
```

The quantized model will show up as checkpoints/$MODEL_REPO/model_int8.pth.

After that we will be using quantized version because of lower memory requirements, but you are free to use original model. To enable it in the example you need to exchange the filename in the [`model_config.yaml`](./model_config.yaml) file.


### Step 2: Generate model archive
At this stage we're creating the model archive which includes the configuration of our model in [model_config.yaml](./model_config.yaml).
It's also the point where we need to decide if we want to deploy our model on a single or multiple GPUs.
For the single GPU case we can use the default configuration that can be found in [model_config.yaml](./model_config.yaml).
All configs enable the current prototyping feature FxGraphCache by setting fx_graph_cache to *true*.
This feature stores the TorchInductor output in a cache to speed up torch.compile times when rerunning the handler.

Please proceed with [TorchServe instalation](https://github.com/pytorch/serve/blob/master/README.md) in order to have torch-model-archiver.

```
torch-model-archiver --model-name gpt_fast_mixtral_moe --version 1.0 --handler ../gpt_fast/handler.py --config-file model_config.yaml --extra-files "gpt-fast/mixtral-moe/generate.py,gpt-fast/mixtral-moe/model.py,gpt-fast/mixtral-moe/quantize.py,gpt-fast/mixtral-moe/tp.py" --archive-format no-archive
mv checkpoints gpt_fast_mixtral_moe/
```

If we want to use tensor parallel variant and split the model over multiple GPUs we need to set the grade of desired tensor parallelism in [model_config_tp.yaml](./model_config_tp.yaml) and use this configuration for creating the archive:
```
torch-model-archiver --model-name gpt_fast_mixtral_moe --version 1.0 --handler ../gpt_fast/handler.py --config-file model_config_tp.yaml --extra-files "gpt-fast/mixtral-moe/generate.py,gpt-fast/mixtral-moe/model.py,gpt-fast/mixtral-moe/quantize.py,gpt-fast/mixtral-moe/tp.py" --archive-format no-archive
mv checkpoints gpt_fast_mixtral_moe/
```

### Step 3: Add the model archive to model store

```
mkdir model_store
mv gpt_fast_mixtral_moe model_store
```

### Step 4: Start torchserve

```
torchserve --start --ncs --model-store model_store --models gpt_fast_mixtral_moe --disable-token-auth  --enable-model-api
```

### Step 5: Run inference

```
curl "http://localhost:8080/predictions/gpt_fast_mixtral_moe" -T request.json
# Returns: Paris, is one of the most visited cities in the world. It is a city of romance, art, culture, and fashion. Paris is home to some of the most iconic landmarks in the world, including the Eiffel Tower
```
