# pylint: disable=W0223
# Details : https://github.com/PyCQA/pylint/issues/3098
"""
Base module for all vision handlers
"""
import base64
import io
from abc import ABC

import torch
from captum.attr import IntegratedGradients
from PIL import Image
from torchvision import transforms

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
        skip_processing = False

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
                image = transforms.ToTensor()(image).to(self.device)
                image = self.image_processing(image)
            else:
                # if the image is a list
                image = torch.FloatTensor(image)
                skip_processing = True

            images.append(image)

        images = torch.stack(images).to(self.device)

        # pre-process images
        # if not skip_processing:
        #    with torch.no_grad():
        #        images = transforms.Normalize(
        #            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        #        )(images)

        return images

    def get_insights(self, tensor_data, _, target=0):
        print("input shape", tensor_data.shape)
        return self.ig.attribute(tensor_data, target=target, n_steps=15).tolist()
