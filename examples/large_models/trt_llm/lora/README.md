# Llama TensorRT-LLM Engine + LoRA model integration with TorchServe

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

## Download Base model & LoRA adapter from Hugging Face
```
huggingface-cli login
# or using an environment variable
huggingface-cli login --token $HUGGINGFACE_TOKEN
```
```
python ../../utils/Download_model.py --model_path model --model_name meta-llama/Meta-Llama-3.1-8B-Instruct --use_auth_token True
python ../../utils/Download_model.py --model_path model --model_name llama-duo/llama3.1-8b-summarize-gpt4o-128k --use_auth_token True
```

## Create TensorRT-LLM Engine
Clone TensorRT-LLM which will be used to create the TensorRT-LLM Engine

```
git clone https://github.com/NVIDIA/TensorRT-LLM.git
```

Compile the model into a TensorRT engine with model weights and a model definition written in the TensorRT-LLM Python API.

```
python TensorRT-LLM/examples/llama/convert_checkpoint.py --model_dir model/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/5206a32e0bd3067aef1ce90f5528ade7d866253f --output_dir ./tllm_checkpoint_1gpu_bf16 --dtype bfloat16
```

```
trtllm-build --checkpoint_dir tllm_checkpoint_1gpu_bf16 --gemm_plugin bfloat16 --gpt_attention_plugin bfloat16 --output_dir ./llama-3.1-8b-engine-lora --max_batch_size 4 --lora_dir model/models--llama-duo--llama3.1-8b-summarize-gpt4o-128k/snapshots/4ba83353f24fa38946625c8cc49bf21c80a22825 --lora_plugin bfloat16
```
If you have enough GPU memory, you can try increasing the `max_batch_size`

You can test if TensorRT-LLM Engine has been compiled correctly by running the following
```
python TensorRT-LLM/examples/run.py --engine_dir ./llama-3.1-8b-engine-lora  --max_output_len 100 --tokenizer_dir model/models--llama-duo--llama3.1-8b-summarize-gpt4o-128k/snapshots/4ba83353f24fa38946625c8cc49bf21c80a22825 --input_text "Amanda: I baked  cookies. Do you want some?\nJerry: Sure \nAmanda: I will bring you tomorrow :-)\n\nSummarize the dialog:" --lora_dir model/models--llama-duo--llama3.1-8b-summarize-gpt4o-128k/snapshots/4ba83353f24fa38946625c8cc49bf21c80a22825 --kv_cache_free_gpu_memory_fraction 0.3 --use_py_session
```
If you are running into OOM, try reducing `kv_cache_free_gpu_memory_fraction`

You should see an output as follows
```
Input [Text 0]: "<|begin_of_text|>Amanda: I baked  cookies. Do you want some?\nJerry: Sure \nAmanda: I will bring you tomorrow :-)\n\nSummarize the dialog:"
Output [Text 0 Beam 0]: " Amanda offered Jerry cookies and said she would bring them to him tomorrow.
Amanda offered Jerry cookies and said she would bring them to him tomorrow.
The dialogue is between Amanda and Jerry. Amanda offers Jerry cookies and says she will bring them to him tomorrow. The dialogue is a simple exchange between two people, with no complex plot or themes. The tone is casual and friendly. The dialogue is a good example of a short, everyday conversation.
The dialogue is a good example of a short,"
```

## Create model archive

```
mkdir model_store
torch-model-archiver --model-name llama3.1-8b --version 1.0 --handler trt_llm_handler --config-file model-config.yaml --archive-format no-archive --export-path model_store -f
mv model model_store/llama3.1-8b/.
mv llama-3.1-8b-engine-lora model_store/llama3.1-8b/.
```

## Start TorchServe
```
torchserve --start --ncs --model-store model_store --models llama3.1-8b --disable-token-auth
```

## Run Inference
```
python ../../utils/test_llm_streaming_response.py -o 50 -t 2 -n 4 -m llama3.1-8b --prompt-text "@prompt.json" --prompt-json
```
