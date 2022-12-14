import base64
import io

import torch
import torch.nn.functional as F
from PIL import Image

from ts.torch_handler.image_classifier import ImageClassifier
from ts.utils.util import map_class_to_label


class NearRealTimeVideoHandler(ImageClassifier):
    def __init__(self):
        super(NearRealTimeVideoHandler, self).__init__()

    def preprocess(self, data):

        images = []
        for row in data:
            imgs = row.get("data") or row.get("body")
            imgs = list(imgs.values())
            for image in imgs:
                if isinstance(image, str):
                    # if the image is a string of bytesarray.
                    image = base64.b64decode(image)

                # If the image is sent as bytesarray
                if isinstance(image, (bytearray, bytes)):
                    image = Image.open(io.BytesIO(image))
                    image = self.image_processing(image)
                else:
                    # if the image is a list
                    image = torch.FloatTensor(image)

                images.append(image)

        return torch.stack(images).to(self.device)

    def postprocess(self, data):

        ps = F.softmax(data, dim=1)
        probs, classes = torch.topk(ps, self.topk, dim=1)
        probs = probs.tolist()
        classes = classes.tolist()

        # Return a list of 1 element of the batch output
        result = []
        output = map_class_to_label(probs, self.mapping, classes)
        result.append(output)
        return result
