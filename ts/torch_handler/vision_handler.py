# pylint: disable=W0223
# Details : https://github.com/PyCQA/pylint/issues/3098
"""
Base module for all vision handlers
"""
from abc import ABC
import io
import torch
from PIL import Image
from .base_handler import BaseHandler


class VisionHandler(BaseHandler, ABC):
    """
    Base class for all vision handlers
    """
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
            if isinstance(image, (bytearray, bytes)):
                image = Image.open(io.BytesIO(image))
                image = self.image_processing(image)
            else:
                image = torch.FloatTensor(image)
            images.append(image)

        return torch.stack(images)
    # def preprocess(self, data):
    #     images = []        
    #     for row in data:
    #         image = row.get("data") or row.get("body")
    #         image = Image.open(io.BytesIO(image))
    #         image = self.image_processing(image)
    #         images.append(image)

    #     return torch.stack(images)
