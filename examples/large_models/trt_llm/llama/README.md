# Llama TensorRT-LLM Engine integration with TorchServe

[TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) provides users with an option to build TensorRT engines for LLMs that contain state-of-the-art optimizations to perform inference efficiently on NVIDIA GPUs.

## Pre-requisites

- TRT-LLM requires Python 3.10
- TRT-LLM works well with python venv (vs conda)
This example is tested with CUDA 12.1
Once TorchServe is installed, install TensorRT-LLM using the following.

```
pip install tensorrt_llm -U --pre --extra-index-url https://pypi.nvidia.com
pip install transformers>=4.44.2
python -c "import tensorrt_llm"
```
shows
```
[TensorRT-LLM] TensorRT-LLM version: 0.13.0.dev2024090300
```

## Download model from HuggingFace
```
huggingface-cli login
# or using an environment variable
huggingface-cli login --token $HUGGINGFACE_TOKEN
```
```
python ../../utils/Download_model.py --model_path model --model_name meta-llama/Meta-Llama-3.1-8B-Instruct --use_auth_token True
```

## Create TensorRT-LLM Engine
Clone TensorRT-LLM which will be used to create the TensorRT-LLM Engine

```
git clone https://github.com/NVIDIA/TensorRT-LLM.git
```

Compile the model into a TensorRT engine with model weights and a model definition written in the TensorRT-LLM Python API.

```
python TensorRT-LLM/examples/llama/convert_checkpoint.py --model_dir model/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/5206a32e0bd3067aef1ce90f5528ade7d866253f/ --output_dir ./tllm_checkpoint_1gpu_bf16 --dtype bfloat16
```

```
trtllm-build --checkpoint_dir tllm_checkpoint_1gpu_bf16 --gemm_plugin bfloat16 --gpt_attention_plugin bfloat16 --max_batch_size 4  --output_dir ./llama-3.1-8b-engine
```
If you have enough GPU memory, you can try increasing the `max_batch_size`

You can test if TensorRT-LLM Engine has been compiled correctly by running the following
```
python TensorRT-LLM/examples/run.py --engine_dir ./llama-3.1-8b-engine  --max_output_len 100 --tokenizer_dir model/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/5206a32e0bd3067aef1ce90f5528ade7d866253f/ --input_text "How do I count to nine in French?"
```
If you are running into OOM, try reducing `kv_cache_free_gpu_memory_fraction`

You should see an output as follows
```
Input [Text 0]: "<|begin_of_text|>How do I count to nine in French?"
Output [Text 0 Beam 0]: " Counting to nine in French is easy and fun. Here's how you can do it:
One: Un
Two: Deux
Three: Trois
Four: Quatre
Five: Cinq
Six: Six
Seven: Sept
Eight: Huit
Nine: Neuf
That's it! You can now count to nine in French. Just remember that the numbers one to five are similar to their English counterparts, but the numbers six to nine have different pronunciations"
```

## Create model archive

```
mkdir model_store
torch-model-archiver --model-name llama3.1-8b --version 1.0 --handler trt_llm_handler --config-file model-config.yaml --archive-format no-archive --export-path model_store -f
mv model model_store/llama3.1-8b/.
mv llama-3.1-8b-engine model_store/llama3.1-8b/.
```

## Start TorchServe
```
torchserve --start --ncs --model-store model_store --models llama3.1-8b --disable-token-auth
```

## Run Inference
```
python ../../utils/test_llm_streaming_response.py -o 50 -t 2 -n 4 -m llama3.1-8b --prompt-text "@prompt.json" --prompt-json
```
