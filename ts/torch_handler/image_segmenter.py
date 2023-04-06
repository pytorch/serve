"""
Module for image segmentation default handler
"""
from torchvision import transforms as T

from ts.handler_utils import ImageSegmentationPostprocess, VisionPreprocess

from .vision_handler import VisionHandler


class ImageSegmenter(VisionHandler):
    """
    ImageSegmenter handler class. This handler takes a batch of images
    and returns output shape as [N K H W],
    where N - batch size, K - number of classes, H - height and W - width.
    """

    image_processing = T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    def __init__(self):
        super().__init__()
        self.preprocess = VisionPreprocess()
        self.postprocess = ImageSegmentationPostprocess()
