# Running Stable diffusion model using Microsoft DeepSpeed-MII in Torchserve.

This document briefs on serving HG Stable diffusion model with Microsoft DeepSpeed-MII in Torchserve. With DeepSpeed-MII there has been significant progress in system optimizations for DL model inference, drastically reducing both latency and cost.

[Model Paper](https://arxiv.org/abs/2112.10752)

### Step 1: Download model

Login into huggingface hub with token by running the below command

```bash
huggingface-cli login
```

paste the token generated from huggingface hub.

```bash
python Download_deepseed_mii_models.py --model_path downloaded_model --model_name CompVis/stable-diffusion-v1-4 --revision main
```

The script prints the path where the model is downloaded as below.

`downloaded_model/models--bert-base-uncased/snapshots/5546055f03398095e385d7dc625e636cc8910bf2/`

#### Run Stable Diffusion model with DeepSpeed-MII

```bash
python deepspeed_mii_stable_diffusion.py --model_path downloaded_model/models--bert-base-uncased/snapshots/5546055f03398095e385d7dc625e636cc8910bf2/ --prompt "a dog chaing a cat"
```

### Step 2: Compress downloaded model

**_NOTE:_** Install Zip cli tool

Navigate to the path got from the above script. Here it is

```bash
cd downloaded_model/models--bert-base-uncased/snapshots/5546055f03398095e385d7dc625e636cc8910bf2/
zip -r /serve/examples/deepspeed_mii/model.zip *
```

### Step 3: Generate MAR file

Navigate up to `deepspeed_mii` directory.

```bash
torch-model-archiver --model-name stable-diffusion --version 1.0 --handler DeepSpeed_mii_handler.py --extra-files model.zip -r requirements.txt
```

DeepSpeed-MII by default support 2 kinds of deployments AzureML and Local deployment. The model optimized by deepspeed MII is served via AzureML endpoint for Azure and gRPC endpoint for local deployment. For Torchserve the internal gRPC server is bye passed and the optimized model in loaded in handler.

**_NOTE:_** Refer `deepspeed_mii_stable_diffusion.py` file for using DeepSpeed-MII without the gRPC server.

[Huggingface Stable Diffusion](https://huggingface.co/blog/stable_diffusion)

### Step 4: Start torchserve

Update config.properties and start torchserve

Increase `max_response_size` for image response.

Refer: https://github.com/pytorch/serve/blob/master/docs/configuration.md#other-properties

```bash
torchserve --start --ts-config config.properties --disable-token-auth  --enable-model-api
```

### Step 5: Run inference

```bash
python query.py --url "http://localhost:8080/predictions/stable-diffusion" --prompt "a photo of an astronaut riding a horse on mars"
```

The image generated will be written to a file `output-20221027213010.jpg`.
