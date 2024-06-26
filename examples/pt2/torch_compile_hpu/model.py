from torchvision.models.resnet import Bottleneck, ResNet


class ImageClassifier(ResNet):
    def __init__(self):
        super(ImageClassifier, self).__init__(Bottleneck, [3, 4, 6, 3])
