import torch
import typer
from pathlib import Path
from .utils import load_model
from .format import Device
import torch.nn.utils.prune

def _prune(model_path : Path, output_name : str = "pruned_model.pt", prune_amount : float = typer.Option(default=0.3, help=" 0 < prune_amount < 1 Percentage of connections to prune"), device : Device = Device.cpu) -> torch.nn.Module:
    """
    Zero out small model weights using l1 norm
    """
    model = load_model(model_path, device)
    
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.LSTM):
            torch.nn.utils.prune.l1_unstructured(module, "weight", prune_amount)
    
    torch.save(model, output_name)
    print("Saved prune model pruned_model.pt")
    return model