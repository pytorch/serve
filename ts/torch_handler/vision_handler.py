# pylint: disable=W0223
# Details : https://github.com/PyCQA/pylint/issues/3098
"""
Base module for all vision handlers
"""
from abc import ABC
import io
import base64
import torch
from PIL import Image
from captum.attr import IntegratedGradients
from .base_handler import BaseHandler


class VisionHandler(BaseHandler, ABC):
    """
    Base class for all vision handlers
    """
    def initialize(self, context):
        super().initialize(context)
        self.ig = IntegratedGradients(self.model)
        self.initialized = True
        properties = context.system_properties
        if not properties.get("limit_max_image_pixels"):
            Image.MAX_IMAGE_PIXELS = None

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
            if isinstance(image, str):
                # if the image is a string of bytesarray.
                image = base64.b64decode(image)

            # If the image is sent as bytesarray
            if isinstance(image, (bytearray, bytes)):
                image = Image.open(io.BytesIO(image))
                image = self.image_processing(image)
            else:
                # if the image is a list
                image = torch.FloatTensor(image)

            images.append(image)

        return torch.stack(images).to(self.device)

    def get_insights(self, tensor_data, _, target=0):
        print("input shape", tensor_data.shape)
        return self.ig.attribute(tensor_data, target=target, n_steps=15).tolist()
