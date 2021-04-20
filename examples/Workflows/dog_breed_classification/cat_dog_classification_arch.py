from torchvision.models.resnet import ResNet, BasicBlock
import torch.nn as nn

class ImageClassifier(ResNet):
    def __init__(self):
        super(ImageClassifier, self).__init__(BasicBlock, [2, 2, 2, 2])
        self.fc = nn.Sequential(nn.Linear(self.fc.in_features, 512), nn.ReLU(), nn.Dropout(), nn.Linear(512, 2))
