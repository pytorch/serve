"""
Module for object detection default handler
"""
from ts.handler_utils import BaseInit, ObjDetectInit, ObjDetectPostprocess

from .vision_handler import VisionHandler


class ObjectDetector(VisionHandler):
    """
    ObjectDetector handler class. This handler takes an image
    and returns list of detected classes and bounding boxes respectively
    """

    def __init__(self):
        super().__init__()
        self.initialize = ObjDetectInit(BaseInit())
        self.postprocess = ObjDetectPostprocess()
