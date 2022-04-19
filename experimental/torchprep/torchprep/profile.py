from pathlib import Path 
from typing import List
from .main import app, Device
from .format import materialize_tensors, parse_input_format
from .utils import profile_model, load_model 

#TODO: Add other profilers like torch profiler or scalene

@app.command()
def profile(model_path : Path, input_shape : Path, iterations : int = 100, device : Device = Device.cpu) -> List[float]:
    """
    Profile model latency 
    """
    if iterations < 100:
        print("Please set iterations > 100")
        return 
    model = load_model(model_path, device)

    input_tensors = materialize_tensors(parse_input_format(input_shape))

    return profile_model(model, input_tensors, iterations)