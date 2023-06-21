# Loading large Huggingface models on Multiple GPUs

This document briefs on serving large HG models on multiple GPUs using deepspeed. We are using facebook/opt-30b in this example

To run this example we need to have deepspeed installed. This has been added to the requirement.txt which can be bundled during model packaging.


```bash
pip install deepspeed

```

### Step 1: Download model

```bash
python ../../utils/Download_model.py --model_path model --model_name facebook/opt-30b --revision main
```

The script prints the path where the model is downloaded as below.

`model/models--facebook--opt-30b/snapshots/ceea0a90ac0f6fae7c2c34bcb40477438c152546`

### Step 2: Generate mar or tgz file

```bash
torch-model-archiver --model-name opt --version 1.0 --handler custom_handler.py --extra-files ds-config.json -r requirements.txt --config-file model-config.yaml --archive-format tgz
```

### Step 3: Add the tgz file to model store

```bash
mkdir model_store
mv opt.tar.gz model_store
```

### Step 4: Start torchserve

```bash
torchserve --start --ncs --model-store model_store --models opt.tar.gz
```

### Step 5: Run inference

```bash
curl -v "http://localhost:8080/predictions/opt" -T sample_text.txt
```

