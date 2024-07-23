# Running Stable diffusion model using Huggingface Diffusers in Torchserve.

### Step 1: Download model

Set access token generated form Huggingface in `Download_model.py` file

Install dependencies

```bash
pip install -r requirements.txt
```

```bash
python Download_model.py
```

### Step 2: Compress downloaded model

**_NOTE:_** Install Zip cli tool

Navigate back to model directory.

```bash

cd Diffusion_model
zip -r ../model.zip *
```

### Step 3: Generate MAR file

Navigate up one level to `diffusers` directory.

```bash
torch-model-archiver --model-name stable-diffusion --version 1.0 --handler stable_diffusion_handler.py --extra-files model.zip -r requirements.txt
```

### Step 4: Start torchserve

Update config.properties and start torchserve

```bash
torchserve --start --ts-config config.properties --disable-token-auth  --enable-model-api
```

### Step 5: Run inference

```bash
python query.py --url "http://localhost:8080/predictions/stable-diffusion" --prompt "a photo of an astronaut riding a horse on mars"
```

The image generated will be written to a file `output-20221027213010.jpg`.

**_NOTE:_** For KServe implementation use the below inputs for v1 and v2 protocols.
Kserve v1 protocol - `sample_v1.json`
Kserve v2 protocol - `sample_v2.json`
