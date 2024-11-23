import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
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
    "--model_name",
    "-m",
    required=True,
    help="HuggingFace model name",
    # type=hf_model,
)

args = parser.parse_args()

pipeline = AutoModelForCausalLM.from_pretrained(
    args.model_name,
    torch_dtype=torch.float16,
    use_safetensors=True,
)
tokenizer = AutoTokenizer.from_pretrained(args.model_name)

pipeline.save_pretrained(args.model_path)
tokenizer.save_pretrained(args.model_path)

print(f"\nFiles for '{args.model_name}' are downloaded to '{args.model_path}'")
