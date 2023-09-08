import io

import torch

from ts.handler_utils.timer import timed
from ts.torch_handler.image_classifier import ImageClassifier

import logging
logger = logging.getLogger(__name__)

class ResNet50Classifier(ImageClassifier):
    """
    ResNet50Classifier handler class. This handler extends class ImageClassifier from image_classifier.py, a
    default handler. This handler takes an image and returns the number in that image.

    Here method preprocess() has been overridden while others are reused from parent class.
    """

    def __init__(self):
        super(ResNet50Classifier, self).__init__()

    @timed
    def preprocess(self, data):
        """The preprocess function of MNIST program converts the input data to a float tensor

        Args:
            data (List): Input data from the request is in the form of a Tensor

        Returns:
            list : The preprocess function returns the input image as a list of float tensors.
        """

        images = []
        for row in data:
            # Compat layer: normally the envelope should just return the data
            # directly, but older versions of Torchserve didn't have envelope.
            image = row.get("data") or row.get("body")
            image = torch.load(io.BytesIO(image))
            images.append(image)

        return torch.stack(images).to(self.device)

