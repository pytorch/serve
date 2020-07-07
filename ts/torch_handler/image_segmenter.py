"""
Module for image segmentation default handler
"""
from torchvision import transforms as T
import torch
import torch.nn.functional as F
from .vision_handler import VisionHandler

class ImageSegmenter(VisionHandler):
    """
    ImageSegmenter handler class. This handler takes a batch of images
    and returns output shape as [N K H W],
    where N - batch size, K - number of classes, H - height and W - width.
    """

    image_processing = T.Compose([T.Resize(256),
                 T.CenterCrop(224),
                 T.ToTensor(),
                 T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])

    def inference(self, data):
        result = super().inference(data)
        return result['out']

class ImangeSegmenter(ImageSegmenter):
    """
    ImageSegmenter was originally misspelled. This is a compat layer
    """
    pass
