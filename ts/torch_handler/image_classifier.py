"""
Module for image classification default handler
"""
import io

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

    TOP_FIVE_CLASSES = 5

    def __init__(self):
        super(ImageClassifier, self).__init__()
        self.topk = ImageClassifier.TOP_FIVE_CLASSES

    def set_max_result_classes(self, topk):
        self.topk = topk

    def get_max_result_classes(self):
        return self.topk

    def preprocess(self, data):
        """
         Scales, crops, and normalizes a PIL image for a PyTorch model,
         returns a tensor
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

        # Convert 2D image to 1D vector
        image = image.unsqueeze(0)

        return image

    def inference(self, data):
        ''' Predict the class (or classes) of an image using a trained deep learning model.
        '''
        inputs = Variable(data).to(self.device)
        outputs = self.model.forward(inputs)
        return outputs

    def postprocess(self, data):
        ps = F.softmax(data, dim=1)
        output_tensor = getattr(ps, self.device.type)()

        # output_tensor size -> [ m, n ] where n is number of clases
        topk = min(self.get_max_result_classes(), list(output_tensor.size())[1])
        topk = output_tensor.topk(topk)

        probs, classes = (e.cpu().data.numpy().squeeze().tolist() for e in topk)

        # handle case when output has only one class
        if not isinstance(probs, list):
            probs = [probs]
            classes = [classes]

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
                results.append({str(classes[index]): str(probs[index])})

        return [results]
