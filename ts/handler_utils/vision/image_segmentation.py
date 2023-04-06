import torch

from ts.handler_utils.caller import PipeCaller


def postprocess(self, data):
    # Returning the class for every pixel makes the response size too big
    # (> 24mb). Instead, we'll only return the top class for each image

    data = data["out"]
    data = torch.nn.functional.softmax(data, dim=1)
    data = torch.max(data, dim=1)
    data = torch.stack([data.indices.type(data.values.dtype), data.values], dim=3)

    return data.tolist()


class ImageSegmentationPostprocess(PipeCaller):
    def __init__(self, previous_handle=None):
        self._prev = previous_handle
        self._method = postprocess
