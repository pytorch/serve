import torch
from torch.export import export
from torchvision.models import ResNet18_Weights, resnet18

# Load the model in eager
model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.eval()

# Need to use pytorch nightlies for using dynamic shapes for dynamic batch sizes
input_data = torch.randn((1, 3, 224, 224))

# Create a torch exported program
exported_program: torch.export.ExportedProgram = export(model, (input_data,))

# Save the exported program in .pt2 format
torch.export.save(exported_program, "resnet18.pt2")
