import torch
import math
import os
from typing import Dict, List
from tqdm import tqdm
from time import time


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

def print_environment_variables() -> Dict[str, str]:
    print(os.environ)

def profile_model(model :torch.nn.Module, input_tensors, label : str = "model", iterations : int = 100) -> List[float]:
    print("Starting profile")

    warmup_iterations = iterations // 10
    for step in range(warmup_iterations):
        model(*input_tensors)

    durations = []
    for step in tqdm(range(iterations)):
        tic = time.time()
        model(*input_tensors)
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