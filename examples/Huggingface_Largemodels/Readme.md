# Loading large Huggingface models with contrained resources using accelerate

This document briefs on serving large HG models with limited resource using accelerate.
### Step 1: Download model

Login into huggingface hub with token by running the below command

```bash
huggingface-cli login
```
paste the token generated from huggingface hub.

```bash
python Download_model.py --model_name bigscience/bloom-7b1
```
The script prints the path where the model is downloaded as below.

`model/models--bigscience-bloom-7b1/snapshots/5546055f03398095e385d7dc625e636cc8910bf2/`

### Step 2: Compress downloaded model

**_NOTE:_** Install Zip cli tool

Navigate to the path got from the above script. Here it is

```bash
cd model/models--bigscience-bloom-7b1/snapshots/5546055f03398095e385d7dc625e636cc8910bf2/
zip -r /serve/examples/deepspeed_mii/model.zip *
```

### Step 3: Generate MAR file

Navigate up to `Huggingface_Largemodels` directory.

```bash
torch-model-archiver --model-name bloom --version 1.0 --handler custom_handler.py --extra-files model.zip,setup_config.json -r requirements.txt
```

### Step 4: Start torchserve

Update config.properties and start torchserve

```bash
torchserve --start --ts-config config.properties
```

### Step 5: Run inference

```bash
python query.py --url "http://localhost:8080/predictions/bloom" -T sample_text.txt
```
