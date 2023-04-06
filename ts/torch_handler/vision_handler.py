# pylint: disable=W0223
# Details : https://github.com/PyCQA/pylint/issues/3098
"""
Base module for all vision handlers
"""
from abc import ABC

from torchvision import transforms

from ts.handler_utils import BaseInitialize, VisionInitialize, VisionPreprocess

from .base_handler import BaseHandler


class VisionHandler(BaseHandler, ABC):
    """
    Base class for all vision handlers
    """

    image_processing = transforms.Compose([transforms.ToTensor()])

    def __init__(self):
        super().__init__()
        self.initialize = VisionInitialize(BaseInitialize())
        self.preprocess = VisionPreprocess()

    def get_insights(self, tensor_data, _, target=0):
        print("input shape", tensor_data.shape)
        return self.ig.attribute(tensor_data, target=target, n_steps=15).tolist()
