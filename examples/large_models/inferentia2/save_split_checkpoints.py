import argparse
import os

import torch
from transformers.models.opt import OPTForCausalLM
from transformers_neuronx.module import save_pretrained_split


def create_directory_if_not_exists(path_str: str) -> str:
    """Creates a directory if it doesn't exist, and returns the directory path."""
    if os.path.isdir(path_str):
        return path_str
    elif input(f"{path_str} does not exist, create directory? [y/n]").lower() == "y":
        os.makedirs(path_str)
        return path_str
    else:
        raise NotADirectoryError(path_str)


def amp_callback(model: OPTForCausalLM, dtype: torch.dtype) -> None:
    """Casts attention and MLP to low precision only; layernorms stay as f32."""
    for block in model.model.decoder.layers:
        block.self_attn.to(dtype)
        block.fc1.to(dtype)
        block.fc2.to(dtype)
    model.lm_head.to(dtype)


# Define and parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_name", "-m", type=str, required=True, help="HuggingFace model name"
)
parser.add_argument(
    "--save_path",
    type=str,
    default="./model-splits",
    help="Output directory for downloaded model files",
)
args = parser.parse_args()

save_path = create_directory_if_not_exists(args.save_path)

# Load HuggingFace model
hf_model = OPTForCausalLM.from_pretrained(args.model_name, low_cpu_mem_usage=True)

# Apply Automatic Mixed Precision (AMP)
amp_callback(hf_model, torch.float16)

# Save the model
save_pretrained_split(hf_model, args.save_path)

print(f"Files for '{args.model_name}' have been downloaded to '{args.save_path}'.")
