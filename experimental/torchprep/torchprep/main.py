import typer
import torch
from typing import List, Union
import time
import os
from enum import Enum
from pathlib import Path
import math
from tqdm import tqdm
import torch.fx as fx
import torch.nn.utils.prune


app = typer.Typer()

class Precision(Enum):
    int8 = "int8"
    float16 = "float16"

class Device(str, Enum):
    cpu = "cpu"
    cuda = "cuda"

class TensorType(str, Enum):
    FLOAT = "float"
    INT   = "int"
    LONG  = "long"

class Architecture(str, Enum):
    ipex = "ipex"
    tensorrt = "tensorrt"
    fastertransformer = "fastertransformer"


@app.command()
def distill(model_path : Path, device : Device = Device.cpu, parameter_scaling : int = 2, layer_scaling : int = None, profile : List[int] = None) -> torch.nn.Module:
    """
    [Coming soon]: Create a smaller student model by setting a distillation ratio and teach it how to behave exactly like your existing model
    """
    print(f"Coming soon")
    print("See this notebook for more information https://colab.research.google.com/drive/1RzQtprrHx8PokLQsFiQPAKzfn_DiTpDN?usp=sharing")

@app.command()
def prune(model_path : Path, prune_amount : float = typer.Option(default=0.3, help=" 0 < prune_amount < 1 Percentage of connections to prune"), device : Device = Device.cpu) -> torch.nn.Module:
    """
    Zero out small model weights using l1 norm
    """
    model = load_model(model_path, device)
    
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.LSTM):
            torch.nn.utils.prune.l1_unstructured(module, "weight", prune_amount)
    
    torch.save(model, 'pruned_model.pt')
    print("Saved prune model pruned_model.pt")
    return model



@app.command()
def fuse(model_path : Path, device : Device = Device.cpu,input_shape : str = typer.Option(default=None, help="Comma seperated input tensor shape e.g 64,3,7,7")) -> torch.nn.Module:
    """
    Supports optimizations including conv/bn fusion, dropout removal and mkl layout optimizations
    Works only for models that are scriptable
    """
    model = load_model(model_path, device)

    if input_shape:
        profile = map(int,input_shape.split(','))
        input_tensor = torch.randn(*profile)
    try:
        model = torch.jit.trace(model,input_shape)
    except Exception as e:
        print(f"{model_path} is not torchscriptable")
        return

    optimized_model = torch.jit.optimize_for_inference(model)

    torch.save(optimized_model, 'optimized_model.pt') 
    return optimized_model

@app.command()
def profile(model_path : Path, iterations : int = 100, device : Device = Device.cpu,
 input_shape : str = typer.Option(default=None, help="Comma seperated input tensor shape e.g 64,3,7,7")) -> List[float]:
    """
    Profile model latency 
    """
    if iterations < 100:
        print("Please set iterations > 100")
        return 
    model = load_model(model_path, device)
    profile = map(int,input_shape.split(','))

    input_tensor = torch.randn(*profile)

    if device == Device.cuda:
        input_tensor.to(torch.device("cuda"))
    return profile_model(model, input_tensor, iterations)

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


@app.command()
def quantize(model_path : Path, precision : Precision ,
 device : Device = Device.cpu, input_shape : str = typer.Option(default=None, help="Comma seperated input tensor shape e.g 64,3,7,7")) -> torch.nn.Module:
    # TODO: define model output path
    # TODO: Support multiple input tensors
    """
    Quantize a saved torch model to a lower precision float format to reduce its size and latency
    """
    model = load_model(model_path, device)

    if device == Device.cpu:
        if precision == "int8":
            dtype = torch.qint8
        else:
            dtype = torch.float16
    quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.LSTM, torch.nn.Linear, torch.nn.Conv2d}, dtype=dtype
)
    # TODO: Add AMP
    if device == Device.cuda:
        if precision == Precision.int8:
            print("int8 precision is not supported for GPUs, defaulting to float16")
        quantized_model = model.half()
    
    print("Model successfully quantized")

    print_size_of_model(model, label = "base model")
    print_size_of_model(quantized_model, label = "quantized_model")
    
    if input_shape:
        profile = map(int,input_shape.split(','))
        input_tensor = torch.randn(*profile)

        if device == Device.cuda:
            input_tensor.to(torch.device("cuda"))
        profile_model(model, input_tensor, label = "base model")
        profile_model(quantized_model, input_tensor, label = "quantized_model")
    
    torch.save(quantized_model, 'quantized_model.pt')
    print(f"model quantized_model.pt was saved")
    return quantized_model


def profile_model(model :torch.nn.Module, input_tensor, label : str = "model", iterations : int = 100) -> List[float]:
    print("Starting profile")

    warmup_iterations = iterations // 10
    for step in range(warmup_iterations):
        model(input_tensor)

    durations = []
    for step in tqdm(range(iterations)):
        tic = time.time()
        model(input_tensor)
        toc = time.time()
        duration = toc - tic
        duration = math.trunc(duration * 1000)
        durations.append(duration)
    avg = sum(durations) / len(durations)
    min_latency = min(durations)
    max_latency = max(durations)
    print(f"Average latency for {label} is: {avg} ms")
    print(f"Min latency for {label} is: {min_latency} ms")
    print(f"Max p99 latency for {label} is: {max_latency} ms")
    return [avg, min_latency, max_latency]

def load_model(model_path: str, device="cpu") -> torch.nn.Module:
    map_location = torch.device(device)
    model = torch.load(model_path, map_location=map_location)
    return model

def print_size_of_model(model : torch.nn.Module, label : str = ""):
    torch.save(model.state_dict(), "temp.p")
    size=os.path.getsize("temp.p")
    print("model: ",label,':','Size (MB):', size/1e6)
    os.remove('temp.p')
    return size