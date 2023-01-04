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
from torchvision import transforms


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
        transform = transforms.Compose([transforms.ToTensor()])
        images = [transform(Image.open(io.BytesIO(base64.b64decode(image)))) if isinstance(image, str) else torch.tensor(image) for image in data]
        return torch.stack(images).to(self.device)

    def get_insights(self, tensor_data, _, target=0):
        print("input shape", tensor_data.shape)
        return self.ig.attribute(tensor_data, target=target, n_steps=15).tolist()
