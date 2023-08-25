# Loading meta-llama/Llama-2-70b-chat-hf on AWS EC2 g5.24xlarge using accelerate

This document briefs on serving large HG models with limited resource using accelerate. This option can be activated with `low_cpu_mem_usage=True`. The model is first created on the Meta device (with empty weights) and the state dict is then loaded inside it (shard by shard in the case of a sharded checkpoint).

### Step 1: Download model Permission

Follow [this instruction](https://huggingface.co/meta-llama/Llama-2-70b-chat-hf) to get permission

Model will be saved in the following path, `model/models--meta-llama--Llama-2-70b-chat-hf/snapshots/9ff8b00464fc439a64bb374769dec3dd627be1c2/`.

### Step 2: Generate MAR file

Add the downloaded path to " model_name:" in `model-config.yaml` and run the following.

```bash
torch-model-archiver --model-name llama2-70b --version 1.0 --handler custom_handler.py --config-file model-config.yaml -r requirements.txt
```

### Step 3: Add the mar file to model store

```bash
mkdir model_store
mv llama2-70b.mar model_store
```

### Step 3: Start torchserve

Update config.properties and start torchserve

```bash
torchserve --start --ncs --ts-config config.properties --model-store model_store --models llama2-70b.mar
```

### Step 4: Run inference

```bash
curl -v "http://localhost:8080/predictions/bloom" -T sample_text.txt
```
