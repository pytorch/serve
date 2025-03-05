# ⚠️ Notice: Limited Maintenance

This project is no longer actively maintained. While existing releases remain available, there are no planned updates, bug fixes, new features, or security patches. Users should be aware that vulnerabilities may not be addressed.

# Serving Llama2 with PyTorch Native Tensor Parallelism

This document briefs on serving the [Llama 2](https://huggingface.co/meta-llama) as presented in the original [Llama repo](https://github.com/facebookresearch/llama/tree/main) using PyTorch(PT) Tensor Parallel (TP) APIs, which under the hood make use of DTensors. It basically, takes a sharding plan for linear layers in MLP and Attention blocks of Llama2 model and make a TP model distributed over multiple GPUs. In the following, we show the steps how to use this and serve the Llama2 7-70B model with Torchserve.

Here we convert the Meta Llama2 model, which is based on Fairscale TP layers to PT distributed compliant checkpoints and use PT TP (DTensor) API to run the Distributed inference.

**Note** The following has been tested on A100 GPUs with 40 GB memory so far.


### How to use it?


1- Make sure you have access to Llama2 weights on [HF model hub](https://huggingface.co/meta-llama), there is a form you need to fill up and within few mins you will get access. Any Llama2 model name on the hub **without -hf** is Meta/FAIR weight.

Make sure you are signed up in HF as well, you will need your API token than can be accessed from [here](https://huggingface.co/settings/tokens), make sure to use the same email for accessing the weights as email you signed in to HF.

Once you have the access, in your terminal login to HF

```
huggingface-cli login YOUR_TOKEN

```

### Step 1: Install requirements

Make sure to have PyTorch Nighlies installed.

```
pip3 install --pre torch  --index-url https://download.pytorch.org/whl/nightly/cu118

pip install transformers fire sentencepiece

```

### Step 2: Download model

Login into HuggingFace hub with token by running the below command, **make sure to specify the right name for the Llama2 model from [HuggingFace (HF) model hub](https://huggingface.co/meta-llama), any model name on the model hub without -hf is Meta original model/ checkpoints and we need them not the HF converted versions.**



```bash
huggingface-cli login
```
paste the token generated from HuggingFace hub. Make sure `use_auth_token=True` is in [Download script](../utils/Download_model.py).

```bash
python ../utils/Download_model.py --model_name meta-llama/Llama-2-7b
```
The script prints the path where the model is downloaded as below.

`model/models--meta-llama--Llama-2-7b/snapshots/365ffa8f1a6c455d3e2028ae658236b4b85ba824`


### Step 3: Convert the "Meta" checkpoints to PyTorch Distributed compliant checkpoints

Convert the checkpoints to  PT-D compliant checkpoints as follows, note that for 7B `--model_parallel_size 1` for 13B would be `--model_parallel_size 2` and 70B `model_parallel_size 8`, you can also set `--nproc_per_node ` accordingly. PT-D compliant support flexible world_size when loading back the checkpoints into TP(lized) model.

You would be able to use larger number of processes/ TP size when load the model back. For example if you have converted the `13B` checkpoints with `--nproc_per_node 2`, during the inference you can use `--nproc_per_node` be `[2, max_num_available_gpu]` which you are changing the world_size and effectively the TP size. The recommendation here is to keep the TP size as shown above respective to model size, 7B (TP Size =1), 13B (TP Size =2), 70B (TP Size =8), unless your benchmark and your batch size/ compute load compensate for communication cost.


This will save the model args in `model_args.json`, during the inference step you need to pass this json file for build the model. Make sure you are setting  `--max_seq_len` which is the maximum sequence length for input text (context length) and `--max_batch_size` which is maximum batch size for inference to respective values. These two values will be used to construct the KV cache.

```
torchrun --nnodes 1 --nproc_per_node 8 convert_checkpoints.py --original_ckpt_dir  PATH/TO/MODEL/CHECKPOINTS  --tokenizer_path PATH/TO/MODEL/CHECKPOINTS/tokenizer.model --model_parallel_size 1 --save_checkpoint_dir converted_checkpoints --max_seq_len 512 --max_batch_size 2

```



### Step 4: set up the configs:

Lets setup configs in `model-config.yaml`

```
#frontend settings
minWorkers: 1
maxWorkers: 1
maxBatchDelay: 200
responseTimeout: 300
parallelType: "tp"
deviceType: "gpu"

torchrun:
    nproc-per-node: 8 # TP size

handler:
    converted_ckpt_dir: "converted_checkpoints"
    tokenizer_path: "tokenizer.model"
    model_args_path: "model_args.json"
    max_seq_len: 512
    max_batch_size: 6
    max_new_tokens: 50
    temperature: 0.6
    top_p: 0.9
    manual_seed: 40
    mode: "text_completion" #choices are text_completion, chat
```

### step 5: Create the mar file:
Create the mar file using the following command here.

```
torch-model-archiver --model-name llama --version 1.0 --handler llama-handler.py --config-file model-config.yaml --archive-format no-archive --extra-files "llama2.py,llama2_tokenizer.py,generate.py,checkpoint_converter.py"

mv converted_checkpoints llama

mv PATH/TO/MODEL/CHECKPOINTS/tokenizer.model llama

mv model_args.json llama

```

### Step 6: Serve the model:

```
torchserve --ncs --start --model-store model_store --models llama

```

### Step 6: Send inference request:

Text completion example :


```bash

curl -v "http://localhost:8080/predictions/llama" -T sample_text.txt

```


Chat example :


```bash

curl -v "http://localhost:8080/predictions/llama" -T dialogs.txt

```
