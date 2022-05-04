import torch
import torch2trt
import ipex
import ort
import os
from enum import Enum
from pathlib import Path
from .main import app, Device, profile
from .format import materialize_tensors, parse_input_format
from .utils import load_model


class Runtime(str, Enum):
    ipex = "ipex"
    tensorrt = "tensorrt"
    fastertransformer = "fastertransformer"

@app.command()
def export_to_runtime(model_path : Path, runtime : Runtime, input_shape : Path, device : Device, output_name : str = "optimized_model.pt"):
    """
    [Not Tested]: Do not use
    Export your model to an optimized runtime for accelerated inference
    """
    model = load_model(model_path)
    input_tensors = materialize_tensors(parse_input_format(input_shape))

    if runtime == Runtime.ipex:
        optimized_model = ipex.optimize(model)
    elif runtime == Runtime.tensorrt:
        optimized_model = torch2trt(model, input_tensors)
    elif runtime == Runtime.ort:
        options = ort.SessionOptions()
        return ort.InferenceSession(model, options)
    
    profile(model, input_shape, device)
    profile(optimized_model, input_shape, device)

    torch.save(optimized_model, output_name)

@app.command()
def env(device : Device = Device.cpu, omp_num_threads : int = 1, kmp_blocktime : int = 1) -> None:
    """
    [Experimental]: Set environment variables for optimized inference. Run this command on the machine where inference will happen!
    """
    if device == Device.cpu:
        os.environ["OMP_NUM_THREADS"] = omp_num_threads
        os.environ["KMP_BLOCKTIME"] = kmp_blocktime
    else:
        print(f"support for architecture {device} coming soon")