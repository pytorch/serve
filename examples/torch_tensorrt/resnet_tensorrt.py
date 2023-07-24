import torch
import torch_tensorrt
from torchvision.models import ResNet50_Weights, resnet50

model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

model.eval()


trt_model_fp16 = torch_tensorrt.compile(
    model,
    inputs=[
        torch_tensorrt.Input(
            min_shape=(1, 3, 224, 224),
            opt_shape=(32, 3, 224, 224),
            max_shape=(64, 3, 224, 224),
            dtype=torch.float32,
        )
    ],
    enabled_precisions=torch.float16,  # Run with FP32
    workspace_size=1 << 22,
)
torch.jit.save(trt_model_fp16, "res50_trt_fp16.pt")
