"""
Module for image classification default handler
"""
import io
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms

from .vision_handler import VisionHandler


class ImageClassifier(VisionHandler):
    """
    ImageClassifier handler class. This handler takes an image
    and returns the name of object in that image.
    """

    def __init__(self):
        super(ImageClassifier, self).__init__()

    def preprocess(self, data):
        """
         Scales, crops, and normalizes a PIL image for a PyTorch model,
         returns an Numpy array
        """
        image = data[0].get("data")
        if image is None:
            image = data[0].get("body")

        my_preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        image = Image.open(io.BytesIO(image))
        image = my_preprocess(image)
        return image

    def inference(self, data):
        ''' Predict the class (or classes) of an image using a trained deep learning model.
        '''
        # Convert 2D image to 1D vector
        topk = 5
        data = np.expand_dims(data, 0)
        data = torch.from_numpy(data)

        inputs = Variable(data).to(self.device)
        outputs = self.model.forward(inputs)

        ps = F.softmax(outputs, dim=1)
        topk = getattr(ps, self.device.type)().topk(topk)

        probs, classes = (e.cpu().data.numpy().squeeze().tolist() for e in topk)

        results = []
        for index, elem in enumerate(probs):
            if self.mapping:
                tmp = dict()
                if isinstance(self.mapping, dict) and isinstance(list(self.mapping.values())[0], list):
                    tmp[self.mapping[str(classes[index])][1]] = elem
                elif isinstance(self.mapping, dict) and isinstance(list(self.mapping.values())[0], str):
                    tmp[self.mapping[str(classes[index])]] = elem
                else:
                    raise Exception('index_to_name mapping should be in "class":"label" json format')

                results.append(tmp)
            else:
                results.append({str(classes[i]):str(probs[i])})

        return [results]

    def postprocess(self, data):
        return data


_service = ImageClassifier()


def handle(data, context):
    """
    Entry point for image classifier default handler
    """
    try:
        if not _service.initialized:
            _service.initialize(context)

        if data is None:
            return None

        data = _service.preprocess(data)
        data = _service.inference(data)
        data = _service.postprocess(data)

        return data
    except Exception as e:
        raise Exception("Please provide a custom handler in the model archive." + e)
