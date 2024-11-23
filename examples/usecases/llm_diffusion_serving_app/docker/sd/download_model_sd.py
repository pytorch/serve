import os
import argparse
import torch
from diffusers import DiffusionPipeline, UNet2DConditionModel
from huggingface_hub import HfApi


def dir_path(path_str):
    try:
        if not os.path.isdir(path_str):
            os.makedirs(path_str)
            print(f"{path_str} did not exist, created the directory.")
            print("\nDownload will take few moments to start.. ")
        return path_str
    except Exception as e:
        raise NotADirectoryError(f"Failed to create directory {path_str}: {e}")


class HFModelNotFoundError(Exception):
    def __init__(self, model_str):
        super().__init__(f"HuggingFace model not found: '{model_str}'")


def hf_model(model_str):
    api = HfApi()
    models = [m.modelId for m in api.list_models()]
    if model_str in models:
        return model_str
    else:
        raise HFModelNotFoundError(model_str)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_path",
    "-o",
    type=dir_path,
    default="model",
    help="Output directory for downloaded model files",
)
parser.add_argument(
    "--model_name", "-m", type=hf_model, required=True, help="HuggingFace model name"
)

args = parser.parse_args()

unet = UNet2DConditionModel.from_pretrained(
    "latent-consistency/lcm-sdxl", torch_dtype=torch.float16, variant="fp16"
)

pipeline = DiffusionPipeline.from_pretrained(
    args.model_name,
    unet=unet,
    torch_dtype=torch.float16,
    variant="fp16",
)
pipeline.save_pretrained(args.model_path)
unet.save_pretrained(f"{args.model_path}/lcm")


print(f"\nFiles for '{args.model_name}' is downloaded to '{args.model_path}'")
