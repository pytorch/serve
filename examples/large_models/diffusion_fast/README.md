
## Diffusion-Fast

[Diffusion fast](https://github.com/huggingface/diffusion-fast) is a simple and efficient pytorch-native way of optimizing Stable Diffusion XL (SDXL).

It features:
* Running with the bfloat16 precision
* scaled_dot_product_attention (SDPA)
* torch.compile
* Combining q,k,v projections for attention computation
* Dynamic int8 quantization

Details about the optimizations and various results can be found in this  [blog](https://pytorch.org/blog/accelerating-generative-ai-3/).
The example has been tested on A10, A100 as well as H100.


#### Pre-requisites

`cd` to the example folder `examples/image_generation/diffusion_fast`

Install dependencies and upgrade torch to nightly build (currently required)
```
git clone https://github.com/huggingface/diffusion-fast.git
pip install accelerate transformers diffusers peft
pip install --no-cache-dir git+https://github.com/pytorch-labs/ao@54bcd5a10d0abbe7b0c045052029257099f83fd9
pip install pandas matplotlib seaborn
```
### Step 1: Download the Stable diffusion model

```bash
python Download_model.py
```
This saves the model in `Base_Diffusion_model`

### Step 1: Generate model archive
At this stage we're creating the model archive which includes the configuration of our model in [model_config.yaml](./model_config.yaml).
It's also the point where we need to decide if we want to deploy our model on a single or multiple GPUs.
For the single GPU case we can use the default configuration that can be found in [model_config.yaml](./model_config.yaml).

```
torch-model-archiver --model-name diffusion_fast --version 1.0 --handler diffusion_fast_handler.py --config-file model_config.yaml --extra-files "diffusion-fast/utils/pipeline_utils.py" --archive-format no-archive
mv Base_Diffusion_model diffusion_fast/
```

### Step 2: Add the model archive to model store

```
mkdir model_store
mv diffusion_fast model_store
```

### Step 3: Start torchserve

```
torchserve --start --ts-config config.properties --model-store model_store --models diffusion_fast --disable-token-auth  --enable-model-api
```

### Step 4: Run inference

```
python query.py --url "http://localhost:8080/predictions/diffusion_fast" --prompt "a photo of an astronaut riding a horse on mars"
```
The image generated will be written to a file `output-<>.jpg`
