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
        super().initialize(context)

        properties = context.system_properties
        # Torchvision breaks with object detector models before 0.6.0
        if version.parse(torchvision_version) < version.parse("0.6.0"):
            self.initialized = False
            self.device = torch.device("cuda" if torch.cuda.is_available() and properties.get("gpu_id") is not None
                                       else "cpu")
            self.model.to(self.device)
            self.model.eval()
            self.initialized = True

    def postprocess(self, data):
        result = []

        box_filters = [row['scores'] >= self.threshold for row in data]
        filtered_boxes, filtered_classes, filtered_scores = [
            [row[key][box_filter].tolist() for row, box_filter in zip(data, box_filters)]
            for key in ['boxes', 'labels', 'scores']
        ]

        for classes, boxes, scores in zip(filtered_classes, filtered_boxes, filtered_scores):
            retval = []
            for _class, _box, _score in zip(classes, boxes, scores):
                _retval = map_class_to_label([[_box]], self.mapping, [[_class]])[0]
                _retval['score'] = _score
                retval.append(_retval)
            result.append(retval)

        return result
