"""
Module for image segmentation default handler
"""
import torch

from .dali_handler import DaliHandler


class DALIImageSegmenter(DaliHandler):
    """
    ImageSegmenter handler class. This handler takes a batch of images
    and returns output shape as [N K H W],
    where N - batch size, K - number of classes, H - height and W - width.
    """

    def postprocess(self, data):
        # Returning the class for every pixel makes the response size too big
        # (> 24mb). Instead, we'll only return the top class for each image
        data = data["out"]
        data = torch.nn.functional.softmax(data, dim=1)
        data = torch.max(data, dim=1)
        data = torch.stack([data.indices.type(data.values.dtype), data.values], dim=3)

        return data.tolist()
