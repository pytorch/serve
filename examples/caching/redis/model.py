from torchvision.models.alexnet import AlexNet


class ImageClassifier(AlexNet):
    def __init__(self):
        super(ImageClassifier, self).__init__()
