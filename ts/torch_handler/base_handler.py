"""
Base default handler to load torchscript or eager mode [state_dict] models
Also, provides handle method per torch serve custom model specification
"""
import abc
import logging
import os
import json
import importlib.util
import pathlib


import torch
from ..utils.util import list_classes_from_module, load_label_mapping

logger = logging.getLogger(__name__)


class BaseHandler(abc.ABC):
    """
    Base default handler to load torchscript or eager mode [state_dict] models
    Also, provides handle method per torch serve custom model specification
    """
    def __init__(self):
        self.model = None
        self.mapping = None
        self.device = None
        self.initialized = False
        self.context = None
        self.manifest = None

    def initialize(self, context):
        """First try to load torchscript else load eager mode state_dict based model"""

        self.manifest = context.manifest
        properties = context.system_properties

        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")

        # Read in the model
        model_dir = properties.get("model_dir")
        serialized_file = self.manifest['model']['serializedFile']
        model_pt_path = os.path.join(model_dir, serialized_file)

        if not os.path.isfile(model_pt_path):
            raise RuntimeError("Missing the model.pt file")

        # Torchscript is better, so try to read that first
        is_torchscript = True
        try:
            logger.debug('Loading torchscript model')
            self.model = self._load_torchscript_model(model_pt_path)
        except Exception as e:
            is_torchscript = False

        if not is_torchscript:
            logger.debug('Torchscript load failed; trying pickle')
            model_file = self.manifest['model']['modelFile']
            self.model = self._load_pickled_model(model_dir, model_file, model_pt_path)

        self.model.to(self.device)
        self.model.eval()

        logger.debug('Model file %s loaded successfully', model_pt_path)

        # Read the mapping file, index to object name
        mapping_file_path = os.path.join(model_dir, "index_to_name.json")

        # Load class mapping for classifiers
        mapping_file_path = os.path.join(model_dir, "index_to_name.json")
        self.mapping = load_label_mapping(mapping_file_path)

        self.initialized = True

    def _load_torchscript_model(self, model_pt_path):
        return torch.jit.load(model_pt_path)

    def _load_pickled_model(self, model_dir, model_file, model_pt_path):
        model_def_path = os.path.join(model_dir, model_file)
        if not os.path.isfile(model_def_path):
            raise RuntimeError("Missing the model.py file")

        module_path = pathlib.PurePath(model_def_path)
        module_name = str(module_path.with_suffix('')).replace('/', '.')
        module_spec = importlib.util.spec_from_file_location(module_name, model_def_path)
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)

        model_class_definitions = list_classes_from_module(module)
        if len(model_class_definitions) != 1:
            raise ValueError("Expected only one class as model definition. {}".format(
                model_class_definitions))

        model_class = model_class_definitions[0]
        state_dict = torch.load(model_pt_path)
        model = model_class()
        model.load_state_dict(state_dict)
        return model



    def preprocess(self, data):
        """
        Override to customize the pre-processing
        :param data: Python list of data items
        :return: input tensor on a device
        """
        return torch.as_tensor(data, device=self.device)

    def inference(self, data):
        """
        Override to customize the inference
        :param data: Torch tensor, matching the model input shape
        :return: Prediction output as Torch tensor
        """
        marshalled_data = data.to(self.device)
        with torch.no_grad():
            results = self.model(marshalled_data)
        return results

    def postprocess(self, data):
        """
        Override to customize the post-processing
        :param data: Torch tensor, containing prediction output from the model
        :return: Python list
        """

        return data.tolist()

    def handle(self, data, context):
        """
        Entry point for default handler
        """

        # It can be used for pre or post processing if needed as additional request
        # information is available in context
        self.context = context

        data = self.preprocess(data)
        data = self.inference(data)
        data = self.postprocess(data)
        return data
