import logging
import os
from collections import Counter

import torch
from torchvision import transforms
from ultralytics import YOLO

from ts.torch_handler.object_detector import ObjectDetector

logger = logging.getLogger(__name__)

try:
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
except ImportError as error:
    XLA_AVAILABLE = False


class Yolov8Handler(ObjectDetector):
    image_processing = transforms.Compose(
        [transforms.Resize(640), transforms.CenterCrop(640), transforms.ToTensor()]
    )

    def __init__(self):
        super(Yolov8Handler, self).__init__()

    def initialize(self, context):
        # Set device type
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif XLA_AVAILABLE:
            self.device = xm.xla_device()
        else:
            self.device = torch.device("cpu")

        # Load the model
        properties = context.system_properties
        self.manifest = context.manifest
        model_dir = properties.get("model_dir")
        self.model_pt_path = None
        if "serializedFile" in self.manifest["model"]:
            serialized_file = self.manifest["model"]["serializedFile"]
            self.model_pt_path = os.path.join(model_dir, serialized_file)
        self.model = self._load_torchscript_model(self.model_pt_path)
        logger.debug("Model file %s loaded successfully", self.model_pt_path)

        self.initialized = True

    def _load_torchscript_model(self, model_pt_path):
        """Loads the PyTorch model and returns the NN model object.

        Args:
            model_pt_path (str): denotes the path of the model file.

        Returns:
            (NN Model Object) : Loads the model object.
        """
        # TODO: remove this method if https://github.com/pytorch/text/issues/1793 gets resolved

        model = YOLO(model_pt_path)
        model.to(self.device)
        return model

    def postprocess(self, res):
        output = []
        for data in res:
            classes = data.boxes.cls.tolist()
            names = data.names

            # Map to class names
            classes = map(lambda cls: names[int(cls)], classes)

            # Get a count of objects detected
            result = Counter(classes)
            output.append(dict(result))

        return output
