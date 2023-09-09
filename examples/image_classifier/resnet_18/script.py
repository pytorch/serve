import torch
from torchvision import models

SCRIPTED_MODEL = "resnet-18.pt"


def create_pt_file(SCRIPTED_MODEL):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    sm = torch.jit.script(model)
    sm.save(SCRIPTED_MODEL)


if __name__ == "__main__":
    create_pt_file(SCRIPTED_MODEL)
