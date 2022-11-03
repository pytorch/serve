import numpy as np
import torch
from torch.profiler import ProfilerActivity
from torchvision import transforms
from ts.torch_handler.image_classifier import ImageClassifier


class MNISTDigitClassifier(ImageClassifier):
    """
    MNISTDigitClassifier handler class. This handler extends class ImageClassifier from image_classifier.py, a
    default handler. This handler takes image as a tensor and returns the number in that image.

    Here method postprocess() has been overridden while others are reused from parent class.
    """

    def __init__(self):
        super(MNISTDigitClassifier, self).__init__()
        self.profiler_args = {
            "activities": [ProfilerActivity.CPU],
            "record_shapes": True,
        }

    def preprocess(self, data):
        """Preprocess the data, fetches the image from the request body and converts to torch tensor.
        Args:
            data (list): Image to be sent to the model for inference.
        Returns:
            tensor: A torch tensor
        """
        return torch.as_tensor(data[0]["body"]["image"])

    def postprocess(self, data):
        """The post process of MNIST converts the predicted output response to a label.

        Args:
            data (list): The predicted output from the Inference with probabilities is passed
            to the post-process function
        Returns:
            list : A list of dictionaries with predictions and explanations is returned
        """
        return data.argmax(1).tolist()
