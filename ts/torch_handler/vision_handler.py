# pylint: disable=W0223
# Details : https://github.com/PyCQA/pylint/issues/3098
"""
Base module for all vision handlers
"""
from abc import ABC

from .base_handler import BaseHandler


class VisionHandler(BaseHandler, ABC):
    """
    Base class for all vision handlers
    """

    def __init__(self):
        super(VisionHandler, self).__init__()
