"""
Module for object detection default handler
"""
import io
import torch
from PIL import Image
from torchvision import transforms
from torchvision import __version__ as torchvision_version
from torch.autograd import Variable
from .vision_handler import VisionHandler


class ObjectDetector(VisionHandler):
    """
    ObjectDetector handler class. This handler takes an image
    and returns list of detected classes and bounding boxes respectively
    """

    def __init__(self):
        super(ObjectDetector, self).__init__()

    def initialize(self, ctx):
        super(ObjectDetector, self).initialize(ctx)
        version = torchvision_version.split(".")

        if int(version[0]) == 0 and int(version[1]) < 6:
            self.initialized = False
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.model.eval()
            self.initialized = True

    def preprocess(self, data):
        """
         Scales, crops, and normalizes a image for a PyTorch model,
         returns an Numpy array
        """
        image = data[0].get("data")
        if image is None:
            image = data[0].get("body")

        my_preprocess = transforms.Compose([transforms.ToTensor()])
        image = Image.open(io.BytesIO(image))
        image = my_preprocess(image)
        return image

    def inference(self, data):
        threshold = 0.5
        # Predict the classes and bounding boxes in an image using a trained deep learning model.
        data = Variable(data).to(self.device)
        pred = self.model([data])  # Pass the image to the model
        if pred[0]['labels'].nelement() == 0:
            return []

        pred_class = list(pred[0]['labels'].cpu().numpy()) # Get the Prediction Score
        # Bounding boxes
        pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].cpu().detach().numpy())]
        pred_score = list(pred[0]['scores'].cpu().detach().numpy())
        # Get list of index with score greater than threshold.
        pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]
        pred_boxes = pred_boxes[:pred_t + 1]
        pred_class = pred_class[:pred_t + 1]
        return [pred_class, pred_boxes]

    def postprocess(self, data):
        if not data:
            return data

        pred_class = data[0]
        try:
            if self.mapping:
                pred_class = [self.mapping['object_type_names'][i] for i in pred_class]  # Get the Prediction Score

            retval = []
            for idx, box in enumerate(data[1]):
                class_name = pred_class[idx]
                retval.append({class_name: str(box)})
            return [retval]
        except Exception as e:
            raise Exception('Object name list file should be json format - {"object_type_names":["person","car"...]}"'
                            + e)


_service = ObjectDetector()


def handle(data, context):
    """
    Entry point for object detector default handler
    """
    try:
        if not _service.initialized:
            _service.initialize(context)

        if data is None:
            return None

        data = _service.preprocess(data)
        data = _service.inference(data)

        if data:
            return _service.postprocess(data)
        else:
            return [[]]
    except Exception as e:
        raise Exception("Please provide a custom handler in the model archive." + e)
