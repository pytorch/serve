from pathlib import Path 
from typing import List
from .format import materialize_tensors, parse_input_format, Device, Profiler
from .utils import profile_model, load_model 
from enum import Enum

def _profile(model_path : Path, input_shape : Path, profiler : Profiler = Profiler.nothing, iterations : int = 100, device : Device = Device.cpu) -> List[float]:
    """
    Profile model latency 
    """
    if iterations < 100:
        print("Please set iterations > 100")
        return 
    model = load_model(model_path, device)

    input_tensors = materialize_tensors(parse_input_format(input_shape))

    return profile_model(model,profiler, input_tensors,model_path,iterations)