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
        images = []

        for row in data:
            image = row.get("data") or row.get("body")
            image = Image.open(io.BytesIO(image))
            image = self.image_processing(image)
            images.append(image)

        return torch.stack(images)
