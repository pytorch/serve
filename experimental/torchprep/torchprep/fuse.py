#TODO: Add helpers with triton for optimized fused ops
import torch
from pathlib import Path
from .main import app, Device
from .utils import load_model


@app.command()
def fuse(model_path : Path, input_shape : Path, output_name : str = "fused_model.pt", device : Device = Device.cpu) -> torch.nn.Module:
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

    torch.save(optimized_model, output_name) 
    return optimized_model