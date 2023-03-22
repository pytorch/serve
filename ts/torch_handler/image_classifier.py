"""
Module for image classification default handler
"""
from torchvision import transforms

from ts.handler_utils import ImageClassificationPostprocess, VisionPreproc

from .vision_handler import VisionHandler


class ImageClassifier(VisionHandler):
    """
    ImageClassifier handler class. This handler takes an image
    and returns the name of object in that image.
    """

    # These are the standard Imagenet dimensions
    # and statistics
    image_processing = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    def __init__(self):
        super().__init__()
        self.preprocess = VisionPreproc()
        self.postprocess = ImageClassificationPostprocess()
