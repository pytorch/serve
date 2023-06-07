# Loading large Huggingface models on Multiple GPUs

This document briefs on serving large HG models on multiple GPUs using deepspeed. To speed up TorchServe regression test, facebook/opt-350m is used in this example. User can choose larger model such as facebook/opt-6.7b.

## Option 1: Using model_dir

### Step 1: Download model

```bash
python ../../utils/Download_model.py --model_path model --model_name facebook/opt-350m --revision main
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

## Option 2: Using model name

### Step 1: Update initialize in custom_handler.py
```python
    def initialize(self, ctx: Context):
        """In this initialize function, the HF large model is loaded and
        partitioned using DeepSpeed.
        Args:
            ctx (context): It is a JSON Object containing information
            pertaining to the model artefacts parameters.
        """
        super().initialize(ctx)
        model_dir = ctx.system_properties.get("model_dir")
        self.max_length = int(ctx.model_yaml_config["handler"]["max_length"])
        self.max_new_tokens = int(ctx.model_yaml_config["handler"]["max_new_tokens"])
        seed = int(ctx.model_yaml_config["handler"]["manual_seed"])
        torch.manual_seed(seed)

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, padding_side="left")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float16)
        self.model.eval()

        ds_engine = get_ds_engine(self.model, ctx)
        self.model = ds_engine.module
        logger.info("Model %s loaded successfully", ctx.model_name)
        self.initialized = True
```

### Step 2: Generate mar or tgz file

```bash
torch-model-archiver --model-name opt --version 1.0 --handler custom_handler.py --extra-files ds-config.json -r requirements.txt --config-file model-config.yaml
```

### Step 3: Add the mar file to model store

```bash
mkdir model_store
mv opt.mar model_store
```

### Step 4: Start torchserve

```bash
torchserve --start --ncs --model-store model_store --models opt.mar
```

### Step 5: Run inference

```bash
curl -v "http://localhost:8080/predictions/opt" -T sample_text.txt
```
