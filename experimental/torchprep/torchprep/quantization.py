import torch 
import typer
from pathlib import Path
from .main import app

from .utils import load_model
from .format import Precision, Device

@app.command()
def quantize(model_path : Path, precision : Precision ,
 output_name : str = "quantized_model.pt",
 device : Device = Device.cpu) -> torch.nn.Module:
    """
    Quantize a saved torch model to a lower precision float format to reduce its size and latency
    """
    model = load_model(model_path, device)

    if device == Device.cpu:
        if precision == "int8":
            dtype = torch.qint8
        elif precision == "float16":
            dtype = torch.float16
        else:
            print("unsupported {dtype}")
            return 

    quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.LSTM, torch.nn.Linear, torch.nn.Conv2d}, dtype=dtype
)
    # TODO: Add AMP
    if device == Device.cuda:
        if precision == Precision.int8:
            print("int8 precision is not supported for GPUs, defaulting to float16")
        quantized_model = model.half()
    
    print("Model successfully quantized")
    
    torch.save(quantized_model, output_name)
    # print(f"model {output_name} was saved")
    return quantized_model
