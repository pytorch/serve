# Loading large Huggingface models with constrained resources using accelerate

This document briefs on serving large HG models with limited resource using accelerate. This option can be activated with `low_cpu_mem_usage=True`. The model is first created on the Meta device (with empty weights) and the state dict is then loaded inside it (shard by shard in the case of a sharded checkpoint).

### Step 1: Set the model checkpoint path in `setup_config.json`

```bash
{
    "revision": "main",
    "max_memory": {
        "0": "40GB",
        "cpu": "10GB"
    },
    "low_cpu_mem_usage": true,
    "device_map": "auto",
    "offload_folder": "offload",
    "offload_state_dict": true,
    "torch_dtype":"float16",
    "max_length":"80",
    "model_path":"PATH/TO/models/13B/"
}
```

### Step 2: Generate MAR file


```bash
torch-model-archiver --model-name llama  --version 1.0 --handler custom_handler.py --extra-files setup_config.json -r requirements.txt
```

**__Note__**: Modifying setup_config.json
- Enable `low_cpu_mem_usage` to use accelerate
- Recommended `max_memory` in `setup_config.json` is the max size of shard.
- Refer: https://huggingface.co/docs/transformers/main_classes/model#large-model-loading

### Step 4: Add the mar file to model store

```bash
mkdir model_store
mv llama.tar.gz model_store
```

### Step 5: Start torchserve

Update config.properties and start torchserve

```bash
torchserve --start --ncs --ts-config config.properties
```

### Step 5: Run inference

```bash
curl -v "http://localhost:8080/predictions/llama" -T sample_text.txt
```





