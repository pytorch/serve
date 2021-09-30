import typer
import torch
from typing import List
import time
import os


app = typer.Typer()


@app.command()
def hello():
    typer.echo(f"Hello torchprep")


@app.command()
def quantize(model_path : str, precision : str , device : str, profile : List[int]):
    model = load_model(model_path, device)

    if device == "cpu":
        if precision == "int8":
            dtype = torch.qint8
        else:
            dtype = torch.float16
    quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.LSTM, torch.nn.Linear, torch.nn.Conv2D}, dtype=dtype
)

        
    if device == "gpu":
        quantized_model = model.half()
        # Save model
    
    print("Model was quantized")

    print_size_of_model(model, label = "base model")
    print_size_of_model(quantized_model, label = "quantized_model")
    
    if profile:
        input_tensor = torch.randn(**profile)
        profile(model, input_tensor, label = "base model")
        profile(quantized_model, input_tensor, label = "quantized_model")


def profile(model, input_tensor, label="model"):
    print("Starting profile")

    durations = []
    for step in range(10):
        tic = time.time()
        model(input_tensor)
        toc = time.time()
        duration = toc - tic
        durations.append(duration)

    print(f"Average latency for {label} is: {sum(durations / len(durations))}")
    print(f"Min latency for {label} is: {min(durations)}")
    print(f"Max latency for {label} is: {max(duration)} ")

def print_size():
    pass

def load_model(model_path: str, device="cpu"):
  map_location = torch.device(device)
  if model_path.endswith(".pt"):
       print("Unsupported Error: Cannot quantize torchscripted models")
  else:
    model = torch.load(model_path, map_location=map_location)
    return model

def print_size_of_model(model, label=""):
    torch.save(model.state_dict(), "temp.p")
    size=os.path.getsize("temp.p")
    print("model: ",label,' \t','Size (MB):', size/1e6)
    os.remove('temp.p')
    return size


if __name__ == "__main__":
    app()