# Serving Huggingface Large Models Fine-tuned with Parameter Efficient Methods (PEFT)

**Parameter Efficient Model Finetuning** 
 This help to make the fine-tuning process much more affordable even on 1 consumer grade GPU. These methods enable us to keep the whole model frozen and just add a tiny learnable parameters/ layers into the model, so technically we just train a very tiny portion of the parameters.The most famous method in this category in [LORA](https://arxiv.org/pdf/2106.09685.pdf), LLaMA Adapter and Prefix-tuning. 

HF [PEFT](https://github.com/huggingface/peft) library provide an easy way of using these methods which we make use of it here. Please read more [here](https://huggingface.co/blog/peft). 

In addition, here we use int8 quantization from [BitandBytes](https://github.com/TimDettmers/bitsandbytes) to ensure lowe latency and memory usage.


Here, we show how to server a PEFT fine-tuned model with Torchserve. In this example, we use `LLaMA 7B` finetuned with `LORA` method.

**Note**

Once you have fine-tuned your model with PEFT library, you will have you PEFT moodule checkpoints save in a directory. We specicy the path to this checkpoints here in the [model_config.yaml](./model-config.yaml) as `peft_model_path`.

## How to serve  HuggingFace PEFT models with Torchserve?

We use a Torchserve custom handler that inherits from base_handler to load the model and define our logic for preprocess, inference and post processing. This is basically very similar to your evaluation process. Following settings used a single A10 GPU.

To run this example we need to have PEFT and Bit&Bytes installed, included in the [requirements.txt](./requirements.txt). This has been added to the requirement.txt which can be bundled during model packaging.


### Step 1: Download model

```bash
python ../utils/Download_model.py --model_name decapoda-research/llama-7b-hf
```
The script prints the path where the model is downloaded as below. This is an example and in your workload you want to use your actual trained model checkpoints.

`model/models--decapoda-research--llama-7b-hf/snapshots/5f98eefcc80e437ef68d457ad7bf167c2c6a1348`




### Step 2: Create a model-config.yaml with that include following

```bash
#frontend settings
minWorkers: 1
maxWorkers: 1
maxBatchDelay: 200
responseTimeout: 300

#backend settings

handler:
    base_model_path: "model/models--decapoda-research--llama-7b-hf/snapshots/5f98eefcc80e437ef68d457ad7bf167c2c6a1348/"
    peft_model_path: "ybelkada/opt-6.7b-lora"
    max_length: 50
    max_new_tokens: 60
    manual_seed: 40
    dtype: fp16

```

### Step 3: Generate Tar/ MAR file

Navigate up to `largemodels` directory. Here as bundling the large model checkpoints is very time consuming, we are passing model checkpoint path in the model_config.yaml as shown above. This let us make the packaging very fast, for production settings, the large models can be put in some shared location and used from there in the model-config.

```bash
torch-model-archiver --model-name llama-peft --version 1.0 --handler pippy_handler.py  -r requirements.txt --config-file model-config.yaml --archive-format tgz

```

### Step 4: Add the mar file to model store

```bash
mkdir model_store
mv llama-peft.tar.gz model_store
```

### Step 5: Start torchserve

Update config.properties and start torchserve

```bash
torchserve --ncs --start --model-store model_store --models llama-peft.tar.gz
```

### Step 6: Run inference

```bash
curl -v "http://localhost:8080/predictions/llama-peft" -T sample_text.txt
```
