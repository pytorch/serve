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

app = typer.Typer()

@app.command()
def distill(model_path : Path, device : Device = Device.cpu, parameter_scaling : int = 2, layer_scaling : int = None) -> torch.nn.Module:
    return _distill(model_path, device, parameter_scaling, layer_scaling)

@app.command()
def fuse(model_path : Path, input_shape : Path, output_name : str = "fused_model.pt", device : Device = Device.cpu) -> Optional[torch.nn.Module]:
    return _fuse(model_path, input_shape, output_name, device)

@app.command()
def prune(model_path : Path, output_name : str = "pruned_model.pt", prune_amount : float = typer.Option(default=0.3, help=" 0 < prune_amount < 1 Percentage of connections to prune"), device : Device = Device.cpu) -> torch.nn.Module:
    return _prune(model_path, output_name, prune_amount, device)

@app.command()
def quantize(model_path : Path, precision : Precision ,output_name : str = "quantized_model.pt", device : Device = Device.cpu) -> torch.nn.Module:
    return _quantize(model_path, precision, output_name, device)

@app.command
def profile(model_path : Path, input_shape : Path, profiler : Profiler = Profiler.nothing, iterations : int = 100, device : Device = Device.cpu) -> List[float]:
    return _profile(model_path, input_shape, profiler, iterations, device)