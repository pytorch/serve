# Loading large Huggingface models with constrained resources using accelerate


This document briefs on serving the [Llama 2](https://huggingface.co/meta-llama) as presented in the original [Llama repo](https://github.com/facebookresearch/llama/tree/main) using FairScale Tensor Parallel. It basically, is relying on Tensor Parallel layers that build the model, more of Megatron Style. In the following, we show the steps how to serve the 7-70B  model with Torchserve.

### Step 1: Download model

Login into huggingface hub with token by running the below command

```bash
huggingface-cli login
```
paste the token generated from huggingface hub. Make sure `use_auth_token=True` is in [Download script](../utils/Download_model.py).

```bash
python ../utils/Download_model.py --model_name meta-llama/Llama-2-7b
```
The script prints the path where the model is downloaded as below.

`model/models--meta-llama--Llama-2-7b/snapshots/365ffa8f1a6c455d3e2028ae658236b4b85ba824`

The downloaded model is around 14GB.

### Step 2: Configure the settings in the model-config.yaml

The current setting has been tested on A100 GPUs, this may not work on GPUs with memories smaller than in A100 for now.

For sering 7B model set the `nproc-per-node: 1`, for 13B `nproc-per-node: 2` and for 70B `nproc-per-node: 8`.

```bash

#frontend settings
minWorkers: 1
maxWorkers: 1
maxBatchDelay: 200
responseTimeout: 300
parallelType: "tp"
deviceType: "gpu"

torchrun:
    nproc-per-node: 1

handler:
    model_path: "PATH/TO/MODEL_CHECKPOINTS"
    tokenizer_path: "PATH/TO/MODEL_CHECKPOINTS/tokenizer.model"
    max_seq_len: 512
    max_batch_size: 6
    max_new_tokens: 60
    temperature: 0.6
    top_p: 0.9
    manual_seed: 40

```


### Step 3: Generate MAR file

```bash
torch-model-archiver --model-name llama --version 1.0 --handler llama-handler.py --config-file model-config.yaml --archive-format tgz -r requirements.txt
```


### Step 4: Add the mar file to model store

```bash
mkdir model_store
mv llama.tar.gz model_store/
```

### Step 5: Start torchserve

Update config.properties and start torchserve

```bash

torchserve --ncs --start --model-store model_store --models llama.tar.gz

```

### Step 5: Run inference

```bash

curl -v "http://localhost:8080/predictions/llama" -T sample_text.txt

```
