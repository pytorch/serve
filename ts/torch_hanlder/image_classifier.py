import io
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms

from .base_handler import BaseHandler


class ImageClassifier(BaseHandler):
    """
    ImageClassifier handler class. This handler takes an image
    and returns the name of object in that image.
    """

    def __init__(self):
        self.checkpoint_file_path = None
        self.mapping = None
        self.initialized = False

    def initialize(self, context):
        """
            Load the model and mapping file to perform infernece.
        """

        # Initialize model
        super(ImageClassifier, self).initialize(context)

        properties = context.system_properties
        model_dir = properties.get("model_dir")

        # Read the mapping file, index to object name
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
