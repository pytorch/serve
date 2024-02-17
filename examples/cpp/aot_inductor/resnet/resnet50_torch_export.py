import os

import torch
from torchvision.models import ResNet50_Weights, resnet50

torch.set_float32_matmul_precision("high")

MAX_BATCH_SIZE = 15

model = resnet50(weights=ResNet50_Weights.DEFAULT)
model.eval()

with torch.no_grad():
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
        # The max batch size is less than 16. The following setting can only work in PT2.3.
        # We need to turn off the below optimizations to support batch_size = 16,
        # which is treated like a special case
        # https://github.com/pytorch/pytorch/pull/116152
        # torch.backends.mkldnn.set_flags(False)
        # torch.backends.nnpack.set_flags(False)

    model = model.to(device=device)
    example_inputs = (torch.randn(2, 3, 224, 224, device=device),)

    batch_dim = torch.export.Dim("batch", min=1, max=MAX_BATCH_SIZE)
    torch._C._GLIBCXX_USE_CXX11_ABI = True
    so_path = torch._export.aot_compile(
        model,
        example_inputs,
        # Specify the first dimension of the input x as dynamic
        dynamic_shapes={"x": {0: batch_dim}},
        # Specify the generated shared library path
        options={
            "aot_inductor.output_path": os.path.join(os.getcwd(), "resnet50_pt2.so"),
            "max_autotune": True,
        },
    )
