# Loading large Huggingface models with PiPPy (PyTorch Native Large inference solution)

This document briefs on serving large HF model with PiPPy.

PiPPy provides pipeline parallelism for serving large models that would not fit into one gpu. It takes your model and splits it into equal sizes (stages) partitioned over the number devices you specify. Then uses micro batching to run your batched input for inference ( its is more optimal for batch sizes >1). Micro-batching is the techniques in pipeline parallelism to maximize gpu utilization.

## How to serve your large HuggingFace models with PiPPy in Torchserve?

We use a Torchserve custom handler that inherits from base_pippy_handler to load the model and define our logic for preprocess, inference and post processing. This is basically very similar to your evaluation process.

### Step 1: Download model

Login into huggingface hub with token by running the below command

```bash
huggingface-cli login
```
paste the token generated from huggingface hub.

```bash
python Download_model.py --model_name facebook/opt-6.7b
```
The script prints the path where the model is downloaded as below. This is an example and in your workload you want to use your actual trained model checkpoints.

`model/models--facebook--opt-6.7b/snapshots/ceea0a90ac0f6fae7c2c34bcb40477438c152546/`

The downloaded model is around 14GB.


### Step 2: Create a model-config.yaml with that include following

```bash

minWorkers: 1
maxWorkers: 1
maxBatchDelay: 100
responseTimeout: 120
parallelLevel: 4
deviceType: "gpu"
parallelType: "pp" #PiPPy as the solution for distributed inference
torchrun:
    nproc-per-node: 4 # specifies the number of processes torchrun starts to serve your model, set to world_size or number of
                       # gpus you wish to split your model
pippy:
    chunks: 1 # This sets the microbatch sizes, microbatch = batch size/ chunks
    input_names: ['input_ids'] # input arg names to the model, this is required for FX tracing
    model_type: "HF" # set the model type to HF if you are using Huggingface model other wise leave it blank or any other model you use.
    rpc_timeout: 1800

handler:
    max_length: 80 # max length of tokens for tokenizer in the handler
```

### Step 3: Generate Tar/ MAR file

Navigate up to `Huggingface_Largemodels` directory.

```bash
torch-model-archiver --model-name opt --version 1.0 --handler pippy_handler.py --extra-files model/models--facebook--opt-6.7b/snapshots/ceea0a90ac0f6fae7c2c34bcb40477438c152546/  -r requirements.txt --config-file model-config.yaml --archive-format tgz

```

### Step 4: Add the mar file to model store

```bash
mkdir model_store
mv opt.tar.gz model_store
```

### Step 5: Start torchserve

Update config.properties and start torchserve

```bash
torchserve --ncs --start --model-store model_store --models opt.tar.gz
```

### Step 6: Run inference

```bash
curl -v "http://localhost:8080/predictions/opt" -T sample_text.txt
```
