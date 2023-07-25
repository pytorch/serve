from abc import ABC

import numpy as np
import torch
from captum.attr import IntegratedGradients
from PIL import Image

from ts.handler_utils.preprocess.dali import get_dali_pipeline
from ts.torch_handler.base_handler import BaseHandler


class DALIHandler(BaseHandler, ABC):
    """
    Base class DeepSpeed handler.
    """

    def initialize(self, context):
        super().initialize(context)
        self.ig = IntegratedGradients(self.model)
        self.initialized = True
        properties = context.system_properties
        if not properties.get("limit_max_image_pixels"):
            Image.MAX_IMAGE_PIXELS = None
        self.pipeline = get_dali_pipeline(context)

    def preprocess(self, data):
        """The preprocess function of MNIST program converts the input data to a float tensor

        Args:
            data (List): Input data from the request is in the form of a Tensor

        Returns:
            list : The preprocess function returns the input image as a list of float tensors.
        """
        images = []
        result = []

        for row in data:
            image = row.get("data") or row.get("body")
            # If the image is sent as bytesarray
            if isinstance(image, (bytearray, bytes)):
                image = np.frombuffer(image, dtype=np.uint8)
            images.append(image)

        response = self.pipeline.run(source=images)
        for idx, _ in enumerate(response[0]):
            data = torch.tensor(response[0].at(idx))
            result.append(data)

        return torch.stack(result).to(self.device)

    def get_insights(self, tensor_data, _, target=0):
        print("input shape", tensor_data.shape)
        return self.ig.attribute(tensor_data, target=target, n_steps=15).tolist()
