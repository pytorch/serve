import rebel
import torch
from torchvision.models import ResNet50_Weights, resnet50

weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)
model.eval()

compiled_model = rebel.compile_from_torch(
    model,
    [("input", [1, 3, 224, 224], torch.float32)],
)
compiled_model.save("resnet50.rbln")
