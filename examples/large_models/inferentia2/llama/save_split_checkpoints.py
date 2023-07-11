import argparse
import os

import torch
from transformers import AutoModelForCausalLM
from transformers_neuronx.module import save_pretrained_split

os.environ["NEURON_CC_FLAGS"] = "--model-type=transformer-inference"


def create_directory_if_not_exists(path_str: str) -> str:
    """Creates a directory if it doesn't exist, and returns the directory path."""
    if os.path.isdir(path_str):
        return path_str
    elif input(f"{path_str} does not exist, create directory? [y/n]").lower() == "y":
        os.makedirs(path_str)
        return path_str
    else:
        raise NotADirectoryError(path_str)


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
hf_model = AutoModelForCausalLM.from_pretrained(
    args.model_name, torch_dtype=torch.float16, device_map="auto"
)

# Save the model
save_pretrained_split(hf_model, args.save_path)

print(f"Files for '{args.model_name}' have been downloaded to '{args.save_path}'.")
