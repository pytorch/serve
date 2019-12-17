from torchvision.models.densenet import DenseNet


class ImageClassifier(DenseNet):
    def __init__(self):
        super(ImageClassifier, self).__init__(48, (6, 12, 36, 24), 96)
