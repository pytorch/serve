import os
import io

import torch
import time
import json
import copy
import numpy as np
import PIL
from PIL import Image
from collections import OrderedDict
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch.nn.functional as F

from .vision_handler import VisionHandler


class ImageClassifier(VisionHandler):
    """
    ImageClassifier service class. This service takes a flower
    image and returns the name of that flower.
    """

    def __init__(self):

        self.checkpoint_file_path = None
        self.model = None
        self.mapping = None
        self.device = "cpu"
        self.initialized = False

    def initialize(self, context):
        """
           Load the model and mapping file to perform infernece.
        """

        properties = context.system_properties
        model_dir = properties.get("model_dir")

        # Read checkpoint file
        checkpoint_file_path = os.path.join(model_dir, "model.pth")
        if not os.path.isfile(checkpoint_file_path):
            raise RuntimeError("Missing model.pth file."+checkpoint_file_path)

            # Prepare the model
        state_dict = torch.load(checkpoint_file_path, map_location='cpu')

        from torchvision.models.densenet import DenseNet
        # Create densnet161 model
        model = DenseNet(48, (6, 12, 36, 24), 96)

        # '.'s are no longer allowed in module names, but previous _DenseLayer
        # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
        # They are also in the checkpoints in model_urls. This pattern is used
        # to find such keys.
        # Credit - https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py#def _load_state_dict()
        import re
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')

        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        model.load_state_dict(state_dict)

        self.model = model

        # Read the mapping file, index to flower
        mapping_file_path = os.path.join(model_dir, "index_to_name.json")
        if not os.path.isfile(mapping_file_path):
            raise RuntimeError("Missing the mapping file")
        with open(mapping_file_path) as f:
            self.mapping = json.load(f)

        self.initialized = True

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

    def inference(self, img, topk=5):
        ''' Predict the class (or classes) of an image using a trained deep learning model.
        '''
        # Convert 2D image to 1D vector
        img = np.expand_dims(img, 0)

        img = torch.from_numpy(img)

        self.model.eval()
        inputs = Variable(img).to(self.device)
        logits = self.model.forward(inputs)

        ps = F.softmax(logits, dim=1)
        topk = ps.cpu().topk(topk)

        probs, classes = (e.data.numpy().squeeze().tolist() for e in topk)

        results = []
        for i in range(len(probs)):
            tmp = dict()
            tmp[self.mapping[str(classes[i])][1]] = probs[i]
            # tmp[self.mapping[str(classes[i])]] = probs[i]
            results.append(tmp)
        return [results]

    def postprocess(self, inference_output):
        return inference_output


# Following code is not necessary if your service class contains `handle(self, data, context)` function
_service = ImageClassifier()


def handle(data, context):
    if not _service.initialized:
        _service.initialize(context)

    if data is None:
        return None

    data = _service.preprocess(data)
    data = _service.inference(data)
    data = _service.postprocess(data)

    return data
