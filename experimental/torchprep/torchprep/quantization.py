import torch 
import typer
from pathlib import Path
from .utils import print_environment_variables, print_size_of_model
from .main import app, Precision, Device

from .format import materialize_tensors, parse_input_format
from .utils import profile_model, load_model

@app.command()
def quantize(model_path : Path, precision : Precision ,
 output_name : str = "quantized_model.pt",
 device : Device = Device.cpu, input_shape : str = typer.Option(default=None, help="Comma seperated input tensor shape e.g 64,3,7,7")) -> torch.nn.Module:
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

    print_size_of_model(model, label = "base model")
    print_size_of_model(quantized_model, label = "quantized_model")
    
    input_tensors = materialize_tensors(parse_input_format(input_shape))
    profile_model(model, input_tensors, label = "base model")
    profile_model(quantized_model, input_tensors, label = "quantized_model")
    
    torch.save(quantized_model, output_name)
    print(f"model {output_name} was saved")
    return quantized_model
