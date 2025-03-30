import torch.nn as nn
from torchvision.models.resnet import BasicBlock, ResNet


class ImageClassifier(ResNet):
    def __init__(self):
        super(ImageClassifier, self).__init__(BasicBlock, [2, 2, 2, 2])
        self.fc = nn.Sequential(
            nn.Linear(self.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, 2),
        )
