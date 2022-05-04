import torch
from pathlib import Path 
from typing import List 
from .format import Device

def _distill(model_path : Path, device : Device = Device.cpu, parameter_scaling : int = 2, layer_scaling : int = None, profile : List[int] = None) -> torch.nn.Module:
    print(f"Coming soon")
    print("See this notebook for more information https://colab.research.google.com/drive/1RzQtprrHx8PokLQsFiQPAKzfn_DiTpDN?usp=sharing")