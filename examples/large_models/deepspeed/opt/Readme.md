# Loading large Huggingface models on Multiple GPUs

This document briefs on serving large HG models on multiple GPUs using deepspeed. To speed up TorchServe regression test, facebook/opt-350m is used in this example. User can choose larger model such as facebook/opt-6.7b.

### Step 1: Download model

Login into huggingface hub with token by running the below command

```bash
huggingface-cli login
```

paste the token generated from huggingface hub.

```bash
python Download_models.py --model_path model --model_name facebook/opt-350m --revision main
```

The script prints the path where the model is downloaded as below.

`model/models--facebook--opt-350m/snapshots/cb32f77e905cccbca1d970436fb0f5e6b58ee3c5/`

### Step 2: Generate mar or tgz file

```bash
torch-model-archiver --model-name opt --version 1.0 --handler custom_handler.py --extra-files model/models--facebook--opt-350m/snapshots/cb32f77e905cccbca1d970436fb0f5e6b58ee3c5/,ds-config.json -r requirements.txt --config-file model-config.yaml --archive-format tgz
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
