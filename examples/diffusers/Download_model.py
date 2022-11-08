import torch
from diffusers import DiffusionPipeline

TOKEN = "Token generated from Huggingface dashboard"

pipeline = DiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    revision="fp16",
    torch_dtype=torch.float16,
    use_auth_token=TOKEN,
)
pipeline.save_pretrained("./Diffusion_model")
