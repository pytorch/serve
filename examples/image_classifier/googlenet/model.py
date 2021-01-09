from torchvision.models.googlenet import GoogLeNet

class ImageClassifier(GoogLeNet):
    def __init__(self):
        super(ImageClassifier, self).__init__()

