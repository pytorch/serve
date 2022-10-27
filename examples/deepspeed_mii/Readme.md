# Running Stable diffusion model using Microsoft DeepSpeed-MII in Torchserve.

### Step 1: Download model

Login into huggingface repo with acesstoken by running the below command

```bash
huggingface-cli login
```
paste the token generated from huggingface hub.

```bash
python Download_deepseed_mii_models.py --model_path downloaded_model --model_name CompVis/stable-diffusion-v1-4 --revision main
```
The script prints the path where the model is downloaded as below.

`downloaded_model/models--bert-base-uncased/snapshots/5546055f03398095e385d7dc625e636cc8910bf2/`

### Step 2: Compress downloaded model

**_NOTE:_** Install Zip cli tool

Navigate to the path got from the above script.

```bash
cd %path-returned-by-above-script%
zip -r /serve/examples/deepspeed_mii/model.zip *
```

### Step 3: Generate MAR file

Navigate up to `deepspeed_mii` directory.

```bash
torch-model-archiver --model-name stable-diffusion --version 1.0 --handler DeepSpeed_mii_handler.py --extra-files model.zip -r requirements.txt
```

### Step 4: Start torchserve

Update config.properties and start torchserve

```bash
torchserve --start --ts-config config.properties
```

### Step 5: Run inference

```bash
python query.py --url "http://localhost:8080/predictions/stable-diffusion" --prompt "a photo of an astronaut riding a horse on mars"
```

The image generated will be written to a file `output-20221027213010.jpg`.
