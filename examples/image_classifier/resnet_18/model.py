from torchvision.models.resnet import ResNet, BasicBlock


class ImageClassifier(ResNet):
    def __init__(self) -> None:
        super(ImageClassifier, self).__init__(BasicBlock, [2, 2, 2, 2])

