# Mixtral Mixture of Experts

This example shows how to deploy [Mixtral-8x7B model](https://huggingface.co/docs/transformers/en/model_doc/mixtral) with HuggingFace with the following features
- `low_cpu_mem_usage=True` for loading with limited resource using `accelerate`
- 8-bit quantization using `bitsandbytes`
- `Accelerated Transformers` using `optimum`
- TorchServe streaming response

## Pre-requisites

Login with a Hugging Face account
```
huggingface-cli login
# or using an environment variable
huggingface-cli login --token $HUGGINGFACE_TOKEN
```

```
python ../Huggingface_accelerate/Download_model.py --model_name mistralai/Mixtral-8x7B-v0.1
```
Model will be saved in the following path, `model/models--mistralai--Mixtral-8x7B-v0.1`.

Install dependencies using

```
pip install -r requirements.txt
```

## Create model archive

```
mkdir model_store
torch-model-archiver --model-name mixtral-moe --version 1.0 --handler hugging_face_llm_handler.py --config-file model-config.yaml --archive-format no-archive --export-path model_store -f
 mv model model_store/mixtral-moe/
 ```

## Start TorchServe

```
torchserve --start --ncs  --model-store model_store --models mixtral-moe
```

Loading large models can take some time. Check the status of the model using

```
curl GET http://localhost:8081/models/mixtral-moe
```
When TorchServe's backend is ready to process inference requests, we should see `"status": "READY"`

```
"workers":
      {
        "id": "9000",
        "startTime": "2024-04-04T17:39:12.079Z",
        "status": "READY",
        "memoryUsage": 8829358080,
        "pid": 4090,
        "gpu": true,
        "gpuUsage": "gpuId::1 utilization.gpu [%]::0 % utilization.memory [%]::0 % memory.used [MiB]::12510 MiB"
      }

```

## Send Inference request

```
python test_streaming.py
```

produces the output

```
What is the difference between cricket and baseball?

- Cricket is a bat-and-ball game played between two teams of eleven players each on a field at the center of which is a rectangular 22-yard-long pitch. Each team takes its turn to bat,
```
