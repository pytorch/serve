# import custom dependency to test that it has been installed
import matplotlib.pyplot as pyplt
from torchvision import transforms

from ts.torch_handler.image_classifier import ImageClassifier


class MNISTDigitClassifier(ImageClassifier):
    image_processing = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    def __init__(self):
        super(MNISTDigitClassifier, self).__init__()

    def postprocess(self, data):
        result = data.argmax(1).tolist()
        # test that custom dependency works
        pyplt.plot(result)
        return result
