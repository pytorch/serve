import io

from PIL import Image
from torchvision import transforms

from ts.torch_handler.image_classifier import ImageClassifier
import base64
import torch

class MNISTDigitClassifier(ImageClassifier):
    """
    MNISTDigitClassifier handler class. This handler extends class ImageClassifier from image_classifier.py, a
    default handler. This handler takes an image and returns the number in that image.

    Here method postprocess() has been overridden while others are reused from parent class.
    """

    image_processing = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    def preprocess(self, data):
        images = []

        for row in data:
            # Compat layer: normally the envelope should just return the data
            # directly, but older versions of Torchserve didn't have envelope.
            if isinstance(row, dict):
                image = row.get("data") or row.get("body") or row
            else:
                image = row

            print("Mnist image code", image)
            images.append(image)

        return torch.stack(images)

    def postprocess(self, data):
        return data.argmax(1).tolist()
