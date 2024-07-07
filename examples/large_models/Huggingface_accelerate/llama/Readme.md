# Loading meta-llama/Meta-Llama-3-70B-Instruct on AWS EC2 g5.24xlarge using accelerate

This document briefs on serving large HF models with limited resource using accelerate. This option can be activated with `low_cpu_mem_usage=True`. The model is first created on the Meta device (with empty weights) and the state dict is then loaded inside it (shard by shard in the case of a sharded checkpoint). This examples uses Meta Llama-3 as an example but it works with Llama2 as well by replacing the model identifier.

### Step 1: Download model Permission

Follow [this instruction](https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct) to get permission

Login with a Hugging Face account
```
huggingface-cli login
# or using an environment variable
huggingface-cli login --token $HUGGINGFACE_TOKEN
```

```bash
python ../Download_model.py --model_path model --model_name meta-llama/Meta-Llama-3-70B-Instruct
```
Model will be saved in the following path, `model/models--meta-llama--Meta-Llama-3-70B-Instruct`.

### Step 2: Generate MAR file

Add the downloaded path to " model_path:" in `model-config.yaml` and run the following.

```bash
torch-model-archiver --model-name llama3-70b-instruct --version 1.0 --handler custom_handler.py --config-file model-config.yaml -r requirements.txt --archive-format no-archive
```

If you are using conda, and notice issues with mpi4py, you can install it with

```
conda install mpi4py
```

### Step 3: Add the mar file to model store

```bash
mkdir model_store
mv llama3-70b-instruct model_store
mv model model_store/llama3-70b-instruct
```

### Step 3: Start torchserve

Update config.properties and start torchserve

```bash
torchserve --start --ncs --ts-config config.properties --model-store model_store --models llama3-70b-instruct --disable-token-auth  --enable-model-api
```

### Step 4: Run inference

```bash
curl -v "http://localhost:8080/predictions/llama3-70b-instruct" -T sample_text.txt
```

results in the following output
```
Mayonnaise is a thick, creamy condiment made from a mixture of egg yolks, oil, vinegar or lemon juice, and seasonings'
```
