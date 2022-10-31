import argparse

import mii
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", "-m", type=str, required=True, help="Model Name")
parser.add_argument(
    "--prompt", "-p", type=str, required=True, help="Input Prompt for image generation"
)
args = parser.parse_args()

provider = mii.constants.ModelProvider["DIFFUSERS"]
config = {"tensor_parallel": 1, "dtype": "fp16"}
mii_configs = mii.MIIConfig(**config)
pipe = mii.models.load_models(
    task_name="text-to-image",
    model_name=args.model_path,
    model_path=args.model_path,
    ds_optimize=False,
    ds_zero=False,
    provider=provider,
    mii_config=mii_configs,
)
pipe.to("cuda")

generator = torch.Generator("cuda").manual_seed(1024)
image = pipe(
    args.prompt,
    guidance_scale=7.5,
    num_inference_steps=50,
    generator=generator,
).images[0]

image.save("output.png")
