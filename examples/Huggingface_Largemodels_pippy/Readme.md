# Loading large Huggingface models with PiPPy (PyTorch Native Large inference solution)

This document briefs on serving large HF model with PiPPy. 

PiPPy provides pipeline paralleism for serving large models that woul not fit into one gpu. It takes your model and splits it into equal sizes (stages) partitioned over the number devices you specify. Then uses micro batching to run your batched input for inference ( its is more optimal for batch sizes >1). Microbatching is the techniques in pipeline parallelsim to maximize gpu utiliztion. 

## How to serve your large HuggingFace models with PiPPY in Torchserve?

We use a Torchserve custom handler that inherits from base_pippy_handler to load the model and define our logic for preprocess, inference and post processing. This is basically very similar to your evaluation process.

### Step 0: Install torchserve from src
```bash
python ts_scripts/install_from_src.py

```
### Step 1: Download model

Login into huggingface hub with token by running the below command

```bash
huggingface-cli login
```
paste the token generated from huggingface hub.

```bash
python Download_model.py --model_name bigscience/bloom-1b1 #facebook/opt-iml-max-1.3b
```
The script prints the path where the model is downloaded as below. This is an example and in your workload you want to use your actual trained model checkpoints. 

`model/models--bigscience-bloom-7b1/snapshots/5546055f03398095e385d7dc625e636cc8910bf2/`

The downloaded model is around 14GB.

### Step 2: Compress downloaded model

**_NOTE:_** Install Zip cli tool

Navigate to the path got from the above script. In this example it is

```bash
cd model/models--bigscience-bloom-7b1/snapshots/5546055f03398095e385d7dc625e636cc8910bf2/
zip -r /home/ubuntu/serve/examples/Huggingface_Largemodels/model.zip *
cd -

```

### Step 3: Generate MAR file

Navigate up to `Huggingface_Largemodels` directory.

```bash
torch-model-archiver --model-name bloom --version 1.0 --handler pippy_handler.py --extra-files model.zip,setup_config.json -r requirements.txt --config-file model-config.yaml
```

**__Note__**: Modifying setup_config.json
- Enable `low_cpu_mem_usage` to use accelerate
- Recommended `max_memory` in `setup_config.json` is the max size of shard.
- Refer: https://huggingface.co/docs/transformers/main_classes/model#large-model-loading

### Step 4: Add the mar file to model store

```bash
mkdir model_store
mv bloom.mar model_store
```

### Step 5: Start torchserve

Update config.properties and start torchserve

```bash
torchserve --ncs --start --model-store model_store --models bloom.mar --ts-config config.properties
```

### Step 5: Run inference

```bash
curl -v "http://localhost:8080/predictions/bloom" -T sample_text.txt
```





