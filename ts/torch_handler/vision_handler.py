from abc import ABC

from .base_handler import BaseHandler


class VisionHandler(BaseHandler, ABC):
    def __init__(self):
        super(VisionHandler, self).__init__()
