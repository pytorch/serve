import os

import torch
from torchvision.models import ResNet18_Weights, resnet18

torch.set_float32_matmul_precision("high")

model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.eval()

with torch.no_grad():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device=device)
    example_inputs = (torch.randn(2, 3, 224, 224, device=device),)
    batch_dim = torch.export.Dim("batch", min=2, max=32)
    so_path = torch._export.aot_compile(
        model,
        example_inputs,
        # Specify the first dimension of the input x as dynamic
        dynamic_shapes={"x": {0: batch_dim}},
        # Specify the generated shared library path
        options={
            "aot_inductor.output_path": os.path.join(os.getcwd(), "resnet18_pt2.so"),
            "max_autotune": True,
        },
    )
