from torchvision.models.squeezenet import SqueezeNet


class ImageClassifier(SqueezeNet):
    def __init__(self):
        super(ImageClassifier, self).__init__('1_1')
