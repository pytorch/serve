import os

import torch
from torchvision.models import ResNet18_Weights, resnet18

torch.set_float32_matmul_precision("high")

model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.eval()

with torch.no_grad():
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
        # The below config is needed for max batch_size = 16
        # https://github.com/pytorch/pytorch/pull/116152
        torch.backends.mkldnn.set_flags(False)
        torch.backends.nnpack.set_flags(False)

    model = model.to(device=device)
    example_inputs = (torch.randn(2, 3, 224, 224, device=device),)

    # Max value is 15 because of https://github.com/pytorch/pytorch/pull/116152
    # On a CUDA enabled device, we tested batch_size of 32.
    batch_dim = torch.export.Dim("batch", min=2, max=15)
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
