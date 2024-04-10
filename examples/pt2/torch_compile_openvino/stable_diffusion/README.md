
## Stable Diffusion XL

[Stable Diffusion XL ](https://huggingface.co/docs/diffusers/en/using-diffusers/sdxl) is a image generation model that is geared towards generating more photorealistic images compared to previous SD models.

The example has been tested on A10, A100 as well as H100.


#### Pre-requisites

`cd` to the example folder `examples/pt2/torch_compile_openvino/stable_diffusion`

### Step 1: Download the Stable diffusion model

```bash
python Download_model.py
```
This saves the model in `Base_Diffusion_model`

### Step 1: Generate model archive
At this stage we're creating the model archive which includes the configuration of our model in [model_config.yaml](./model_config.yaml).

### Enable OpenVINO backend
To use torch.compile with OpenVINO backend, add the following line to the model_config.yaml file

```
pt2 : {backend: "openvino", options: {"aot_autograd": True}}
```

We can also add additional options like model_caching, device and many more OpenVINO specific configs as well in options. For more information, refer to <--TODO: ADD LINK-->

#### For Example:
```
options: {"model_caching" : True, "device": "gpu"}
```

It's also the point where we need to decide if we want to deploy our model on a single or multiple GPUs.
For the single GPU case we can use the default configuration that can be found in [model_config.yaml](./model_config.yaml).

```
torch-model-archiver --model-name diffusion_fast --version 1.0 --handler stable_diffusion_handler.py --config-file model_config.yaml --extra-files "./pipeline_utils.py" --archive-format no-archive
mv Base_Diffusion_model diffusion_fast/
```

### Step 2: Add the model archive to model store

```
mkdir model_store
mv diffusion_fast model_store
```

### Step 3: Start torchserve

```
torchserve --start --ts-config config.properties --model-store model_store --models diffusion_fast
```

### Step 4: Run inference

```
python query.py --url "http://localhost:8080/predictions/diffusion_fast" --prompt "a photo of an astronaut riding a horse on mars"
```
The image generated will be written to a file `output-<>.jpg`
