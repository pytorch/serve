"""
Base default handler to load torchscript or eager mode [state_dict] models
Also, provides handle method per torch serve custom model specification
"""
import abc
import logging
import os
import json
import importlib

import torch
from ..utils.util import list_classes_from_module

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
        self.manifest = None

    def initialize(self, ctx):
        """First try to load torchscript else load eager mode state_dict based model"""

        self.manifest = ctx.manifest

        properties = ctx.system_properties
        model_dir = properties.get("model_dir")
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")

        # model serialize/pt file
        serialized_file = self.manifest['model']['serializedFile']
        model_pt_path = os.path.join(model_dir, serialized_file)
        if not os.path.isfile(model_pt_path):
            raise RuntimeError("Missing the model.pt file")

        map_location = 'cuda' if torch.cuda.is_available() else 'cpu'
        if 'modelFile' in self.manifest['model']:
            model_file = self.manifest['model']['modelFile']
            module = importlib.import_module(model_file.split(".")[0])
            model_class_definitions = list_classes_from_module(module)
            if len(model_class_definitions) != 1:
                raise ValueError("Expected only one class as model definition. {}".format(
                    model_class_definitions))

            model_class = model_class_definitions[0]
            state_dict = torch.load(model_pt_path, map_location=map_location)
            self.model = model_class()
            self.model.load_state_dict(state_dict)
        else:
            logger.debug('No model file found for eager mode, trying to load torchscript model')
            self.model = torch.jit.load(model_pt_path, map_location=map_location)

        self.model.to(self.device)
        self.model.eval()

        logger.debug('Model file %s loaded successfully', model_pt_path)
        # Read the mapping file, index to object name
        mapping_file_path = os.path.join(model_dir, "index_to_name.json")

        if os.path.isfile(mapping_file_path):
            with open(mapping_file_path) as f:
                self.mapping = json.load(f)
        else:
            logger.warning('Missing the index_to_name.json file. Inference output will not include class name.')

        self.initialized = True

    @abc.abstractmethod
    def preprocess(self, data):
        pass

    @abc.abstractmethod
    def inference(self, data):
        pass

    @abc.abstractmethod
    def postprocess(self, data):
        pass
