import argparse
import os

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.models.opt import OPTForCausalLM
from transformers_neuronx.llama.model import LlamaForSampling
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


def opt_amp_callback(model: OPTForCausalLM, dtype: torch.dtype) -> None:
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
subparsers = parser.add_subparsers(dest="subparser")
parser_neuron_cache = subparsers.add_parser("generate_neuron_cache")
parser_neuron_cache.add_argument(
    "--neuron_cache_dir",
    type=str,
    required=True,
    help="Target directory to store neuronx-cc compiled model",
)
parser_neuron_cache.add_argument(
    "--batch_size", type=int, required=True, help="Batch size for the compiled model"
)
parser_neuron_cache.add_argument(
    "--amp", type=str, required=True, help="Automatic mixed precision"
)
parser_neuron_cache.add_argument(
    "--tp_degree",
    type=int,
    required=True,
    help="Tensor parallelism degree for the compiled model",
)
args = parser.parse_args()

save_path = create_directory_if_not_exists(args.save_path)

# Load HuggingFace model config
hf_model_config = AutoConfig.from_pretrained(args.model_name)

# Load HuggingFace model
hf_model = AutoModelForCausalLM.from_pretrained(args.model_name, low_cpu_mem_usage=True)

# Apply Automatic Mixed Precision (AMP)
if hf_model_config.model_type == "opt":
    opt_amp_callback(hf_model, torch.float16)

# Save the model
save_pretrained_split(hf_model, args.save_path)

# Load and save tokenizer for the model
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
tokenizer.save_pretrained(args.save_path)

print(f"Files for '{args.model_name}' have been downloaded to '{args.save_path}'.")

if args.subparser == "generate_neuron_cache":
    os.environ["NEURONX_CACHE"] = "on"
    os.environ["NEURONX_DUMP_TO"] = create_directory_if_not_exists(
        args.neuron_cache_dir
    )
    os.environ["NEURON_CC_FLAGS"] = "--model-type=transformer-inference"

    if hf_model_config.model_type == "llama":
        model = LlamaForSampling.from_pretrained(
            args.save_path,
            batch_size=args.batch_size,
            amp=args.amp,
            tp_degree=args.tp_degree,
        )
    else:
        raise RuntimeError(
            f"Neuron cache generation for model {args.model_name} not supported"
        )

    print(f"Compiling '{args.model_name}'")
    model.to_neuron()
    print(f"Neuron cache for '{args.model_name}' saved to {args.neuron_cache_dir}")
