# Loading large Huggingface models on Multiple GPUs

This document briefs on serving large HuggingFace (HF) models on multiple GPUs using deepspeed. We are using facebook/opt-30b in this example

### Pre-requisites

- Install CUDA. Verified to be working with CUDA 11.7.
- Verified to be working with:

```bash
torch                   2.0.1+cu117
torch-model-archiver    0.8.2
torch-workflow-archiver 0.2.10
torchaudio              2.0.2+cu117
torchdata               0.6.1
torchserve              0.8.2
torchtext               0.15.2+cpu
torchvision             0.15.2+cu117
transformers            4.33.1
deepspeed               0.10.2
```

To run this example we need to have deepspeed installed. This has been added to the requirement.txt which can be bundled during model packaging.

```bash
pip install deepspeed

```

### Step 1: Download model

```bash
python ../utils/Download_model.py --model_path model --model_name facebook/opt-30b --revision main
```

The script prints the path where the model is downloaded as below.

`opt/model/models--facebook--opt-30b/snapshots/ceea0a90ac0f6fae7c2c34bcb40477438c152546`

### Step 2: Generate mar or tgz file

```bash
torch-model-archiver --model-name opt --version 1.0 --handler custom_handler.py --extra-files ds-config.json -r requirements.txt --config-file opt/model-config.yaml --archive-format tgz
```

### Step 3: Add the tgz file to model store

```bash
mkdir model_store
mv opt.tar.gz model_store
```

### Step 4: Start torchserve

```bash
torchserve --start --ncs --model-store model_store --models opt.tar.gz --disable-token-auth  --enable-model-api
```

### Step 5: Run inference

```bash
curl  "http://localhost:8080/predictions/opt" -T sample_text.txt
```
