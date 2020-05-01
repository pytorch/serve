import io
import logging
import numpy as np
import os
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms

logger = logging.getLogger(__name__)


class BatchImageClassifier(object):
    """
    BatchImageClassifier handler class. This handler takes list of images
    and returns a corresponding list of classes
    """

    def __init__(self):
        self.model = None
        self.mapping = None
        self.device = None
        self.initialized = False

    def initialize(self, context):
        """First try to load torchscript else load eager mode state_dict based model"""

        self.manifest = context.manifest
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")

        # Read model serialize/pt file
        serialized_file = self.manifest['model']['serializedFile']
        model_pt_path = os.path.join(model_dir, serialized_file)
        if not os.path.isfile(model_pt_path):
            raise RuntimeError("Missing the model.pt file")

        try:
            logger.info('Loading torchscript model to device {}'.format(self.device))
            self.model = torch.jit.load(model_pt_path)
        except Exception as e:
            # Read model definition file
            model_file = self.manifest['model']['modelFile']
            model_def_path = os.path.join(model_dir, model_file)
            if not os.path.isfile(model_def_path):
                raise RuntimeError("Missing the model.py file")

            state_dict = torch.load(model_pt_path)
            from model import ResNet152ImageClassifier
            self.model = ResNet152ImageClassifier()
            self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        logger.debug('Model file {0} loaded successfully'.format(model_pt_path))

        # Read the mapping file, index to object name
        mapping_file_path = os.path.join(model_dir, "index_to_name.json")
        import json
        if os.path.isfile(mapping_file_path):
            with open(mapping_file_path) as f:
                self.mapping = json.load(f)
        else:
            logger.warning('Missing the index_to_name.json file. Inference output will not include class name.')

        self.initialized = True

    def preprocess(self, request):
        """
         Scales, crops, and normalizes a PIL image for a PyTorch model,
         returns an Numpy array
        """

        image_tensor = None

        for idx, data in enumerate(request):
            image = data.get("data")
            if image is None:
                image = data.get("body")

            my_preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
            input_image = Image.open(io.BytesIO(image))
            input_image = my_preprocess(input_image).unsqueeze(0)
            input_image = Variable(input_image).to(self.device)
            if input_image.shape is not None:
                if image_tensor is None:
                    image_tensor = input_image
                else:
                    image_tensor = torch.cat((image_tensor, input_image), 0)

        return image_tensor

    def inference(self, img):
        return self.model.forward(img)

    def postprocess(self, inference_output):
        num_rows, num_cols = inference_output.shape
        output_classes = []
        for i in range(num_rows):
            out = inference_output[i].unsqueeze(0)
            _, y_hat = out.max(1)
            predicted_idx = str(y_hat.item())
            output_classes.append(self.mapping[predicted_idx])
        return output_classes


_service = BatchImageClassifier()


def handle(data, context):
    if not _service.initialized:
        _service.initialize(context)

    if data is None:
        return None

    data = _service.preprocess(data)
    data = _service.inference(data)
    data = _service.postprocess(data)

    return data
