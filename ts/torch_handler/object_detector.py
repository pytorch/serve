"""
Module for object detection default handler
"""
import torch
from torchvision import transforms
from torchvision import __version__ as torchvision_version
from packaging import version
from .vision_handler import VisionHandler
from ..utils.util import map_class_to_label

class ObjectDetector(VisionHandler):
    """
    ObjectDetector handler class. This handler takes an image
    and returns list of detected classes and bounding boxes respectively
    """

    image_processing = transforms.Compose([transforms.ToTensor()])
    threshold = 0.5

    def initialize(self, context):
        super(ObjectDetector, self).initialize(context)

        # Torchvision breaks with object detector models before 0.6.0
        if version.parse(torchvision_version) < version.parse("0.6.0"):
            self.initialized = False
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.model.eval()
            self.initialized = True

    def postprocess(self, data):
        box_filters = [row['scores'] >= self.threshold for row in data]
        filtered_boxes, filtered_classes, filtered_scores = [
            [row[key][box_filter].tolist() for row, box_filter in zip(data, box_filters)]
            for key in ['boxes', 'labels', 'scores']
        ]

        retval = []
        for classes, box, score in zip(filtered_classes[0], filtered_boxes[0], filtered_scores[0]):
            _retval = map_class_to_label([[box]], self.mapping, [[classes]])[0]
            _retval['score'] = score
            retval.append(_retval)

        return [retval]
