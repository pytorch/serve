from torchvision.models.resnet import ResNet, Bottleneck


class RestNet152ImageClassifier(ResNet):
    def __init__(self):
        super(RestNet152ImageClassifier, self).__init__(Bottleneck, [3, 8, 36, 3])