import typer
from pathlib import Path
import torch
from .format import Device, Profiler, Precision 
from typing import List, Optional

from .distillation import _distill
from .fusion import _fuse
from .profile import _profile
from .quantization import _quantize
from .pruning import _prune
from .runtime import _export_to_runtime, Runtime, _env

app = typer.Typer()

@app.command()
def distill(model_path : Path, device : Device = Device.cpu, parameter_scaling : int = 2, layer_scaling : int = None) -> torch.nn.Module:
    """
    [Coming soon]: Create a smaller student model by setting a distillation ratio and teach it how to behave exactly like your existing model
    """
    return _distill(model_path, device, parameter_scaling, layer_scaling)

@app.command()
def fuse(model_path : Path, input_shape : Path, output_name : str = "fused_model.pt", device : Device = Device.cpu) -> Optional[torch.nn.Module]:
    """
    Supports optimizations including conv/bn fusion, dropout removal and mkl layout optimizations
    Works only for models that are scriptable
    """
    return _fuse(model_path, input_shape, output_name, device)

@app.command()
def prune(model_path : Path, output_name : str = "pruned_model.pt", prune_amount : float = typer.Option(default=0.3, help=" 0 < prune_amount < 1 Percentage of connections to prune"), device : Device = Device.cpu) -> torch.nn.Module:
    """
    Zero out small model weights using l1 norm
    """
    return _prune(model_path, output_name, prune_amount, device)

@app.command()
def quantize(model_path : Path, precision : Precision ,output_name : str = "quantized_model.pt", device : Device = Device.cpu) -> torch.nn.Module:
    """
    Quantize a saved torch model to a lower precision float format to reduce its size and latency
    """
    return _quantize(model_path, precision, output_name, device)

@app.command
def profile(model_path : Path, input_shape : Path, profiler : Profiler = Profiler.nothing, iterations : int = 100, device : Device = Device.cpu) -> List[float]:
    """
    Profile model latency given an input yaml file
    """
    return _profile(model_path, input_shape, profiler, iterations, device)

@app.command()
def export_to_runtime(model_path : Path, runtime : Runtime, input_shape : Path, device : Device, output_name : str = "optimized_model.pt"):
    """
    [Not Tested]: Do not use
    Export your model to an optimized runtime for accelerated inference
    """
    return _export_to_runtime(model_path, runtime, input_shape, device, output_name)

@app.command()
def env(device : Device = Device.cpu, omp_num_threads : int = 1, kmp_blocktime : int = 1) -> None:
    """
    Set optimized environment variables
    """
    return _env(device, omp_num_threads, kmp_blocktime)