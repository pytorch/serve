"""
Module for image classification default handler
"""
import inspect
import logging
import os
import importlib.util
import time
import io
import torch

logger = logging.getLogger(__name__)


class DenseNetHandler:
    """
    DenseNetHandler handler class. This handler takes an image
    and returns the name of object in that image.
    """

    def __init__(self):
        self.model = None
        self.device = None
        self.initialized = False
        self.context = None
        self.manifest = None
        self.map_location = None

    def initialize(self, context):
        """First try to load torchscript else load eager mode state_dict based model"""

        properties = context.system_properties
        self.map_location = "cuda" if torch.cuda.is_available() and properties.get("gpu_id") is not None else "cpu"
        self.device = torch.device(
            self.map_location + ":" + str(properties.get("gpu_id"))
            if torch.cuda.is_available() and properties.get("gpu_id") is not None
            else self.map_location
        )
        self.manifest = context.manifest

        model_dir = properties.get("model_dir")
        serialized_file = self.manifest["model"]["serializedFile"]
        model_pt_path = os.path.join(model_dir, serialized_file)

        if not os.path.isfile(model_pt_path):
            raise RuntimeError("Missing the model.pt file")

        # model def file
        model_file = self.manifest["model"].get("modelFile", "")

        if model_file:
            logger.debug("Loading eager model")
            self.model = self._load_pickled_model(model_dir, model_file, model_pt_path)
        else:
            logger.debug("Loading torchscript model")
            self.model = self._load_torchscript_model(model_pt_path)

        self.model.to(self.device)
        self.model.eval()

        logger.debug("Model file %s loaded successfully", model_pt_path)

        self.initialized = True

    def _load_torchscript_model(self, model_pt_path):
        return torch.jit.load(model_pt_path, map_location=self.map_location)

    def _load_pickled_model(self, model_dir, model_file, model_pt_path):
        model_def_path = os.path.join(model_dir, model_file)
        if not os.path.isfile(model_def_path):
            raise RuntimeError("Missing the model.py file")

        module = importlib.import_module(model_file.split(".")[0])
        model_class_definitions = list_classes_from_module(module)
        if len(model_class_definitions) != 1:
            raise ValueError(
                "Expected only one class as model definition. {}".format(
                    model_class_definitions
                )
            )

        model_class = model_class_definitions[0]
        state_dict = torch.load(model_pt_path, map_location=self.map_location)
        model = model_class()
        model.load_state_dict(state_dict)
        return model

    def inference(self, data, *args, **kwargs):
        """
        Override to customize the inference
        :param data: Torch tensor, matching the model input shape
        :return: Prediction output as Torch tensor
        """
        marshalled_data = data.to(self.device)
        with torch.no_grad():
            results = self.model(marshalled_data, *args, **kwargs)
        return results

    def handle(self, data, context):
        """
        Entry point for default handler
        """

        # It can be used for pre or post processing if needed as additional request
        # information is available in context
        start_time = time.time()

        self.context = context
        metrics = self.context.metrics

        values = []
        for row in data:
            image = row.get("data") or row.get("body")
            tensor = torch.load(io.BytesIO(image))
            values.append(tensor)
        data = self.inference(torch.stack(values))

        stop_time = time.time()
        metrics.add_time(
            "HandlerTime", round((stop_time - start_time) * 1000, 2), None, "ms"
        )
        return [data]


def list_classes_from_module(module, parent_class=None):
    """
    Parse user defined module to get all model service classes in it.

    :param module:
    :param parent_class:
    :return: List of model service class definitions
    """

    # Parsing the module to get all defined classes
    classes = [
        cls[1]
        for cls in inspect.getmembers(
            module,
            lambda member: inspect.isclass(member)
            and member.__module__ == module.__name__,
        )
    ]
    # filter classes that is subclass of parent_class
    if parent_class is not None:
        return [c for c in classes if issubclass(c, parent_class)]

    return classes
