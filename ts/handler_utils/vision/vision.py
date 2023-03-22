import base64
import io

import torch
from captum.attr import IntegratedGradients
from PIL import Image

from ts.handler_utils.caller import InitCaller, PipeCaller


def vision_preprocess(obj, data):
    """The preprocess function of MNIST program converts the input data to a float tensor

    Args:
        data (List): Input data from the request is in the form of a Tensor

    Returns:
        list : The preprocess function returns the input image as a list of float tensors.
    """
    images = []

    for row in data:
        # Compat layer: normally the envelope should just return the data
        # directly, but older versions of Torchserve didn't have envelope.
        image = row.get("data") or row.get("body")
        if isinstance(image, str):
            # if the image is a string of bytesarray.
            image = base64.b64decode(image)

        # If the image is sent as bytesarray
        if isinstance(image, (bytearray, bytes)):
            image = Image.open(io.BytesIO(image))
            image = obj.image_processing(image)
        else:
            # if the image is a list
            image = torch.FloatTensor(image)

        images.append(image)

    return torch.stack(images).to(obj.device)

    # pylint: enable=unnecessary-pass


def vision_initialize(obj, context):
    obj.ig = IntegratedGradients(obj.model)
    obj.initialized = True
    properties = context.system_properties
    if not properties.get("limit_max_image_pixels"):
        Image.MAX_IMAGE_PIXELS = None


class VisionPreprocess(PipeCaller):
    def __init__(self, previous_handle=None, image_processing=None):
        self._prev = previous_handle
        self._method = vision_preprocess


class VisionInitialize(InitCaller):
    def __init__(self, previous_handle=None):
        self._prev = previous_handle
        self._method = vision_initialize
