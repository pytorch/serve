from torchvision.models.alexnet import AlexNet


class ImageClassifier(AlexNet):
    def __init__(self) -> None:
        super(ImageClassifier, self).__init__()
