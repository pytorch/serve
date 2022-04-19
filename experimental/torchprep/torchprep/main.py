import typer
import torch
from typing import List
import os
from enum import Enum
from pathlib import Path
import torch.fx as fx
import torch.nn.utils.prune
from .format import parse_input_format, materialize_tensors
from .utils import load_model, profile_model, print_size_of_model, print_environment_variables


app = typer.Typer()


# TODO: Need to remove these 2 enums
class Precision(Enum):
    int8 = "int8"
    float16 = "float16"

class Device(str, Enum):
    cpu = "cpu"
    cuda = "cuda"