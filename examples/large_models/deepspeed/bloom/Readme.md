# Loading large Huggingface models with constrained resources using accelerate

This document briefs on serving large HG models with limited resource using deepspeed.

### Step 1: Generate mar or tgz file

```bash
torch-model-archiver --model-name bloom --version 1.0 --handler custom_handler.py --extra-files ds-config.json -r requirements.txt --config-file model-config.yaml --archive-format tgz
```

### Step 2: Add the tgz file to model store

```bash
mkdir model_store
mv bloom.tar.gz model_store
```

### Step 3: Start torchserve


```bash
torchserve --start --ncs --ts-config config.properties
```

### Step 4: Run inference

```bash
curl -v "http://localhost:8080/predictions/bloom" -T sample_text.txt
```
