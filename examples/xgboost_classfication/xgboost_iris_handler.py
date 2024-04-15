import logging
import os

import numpy as np
import torch
from xgboost import XGBClassifier

from ts.torch_handler.base_handler import BaseHandler
from ts.utils.util import load_label_mapping

logger = logging.getLogger(__name__)


class XGBIrisHandler(BaseHandler):
    def __init__(self):
        super().__init__()

    def initialize(self, context):
        # Set device type
        self.device = torch.device("cpu")

        # Load the model
        properties = context.system_properties
        self.manifest = context.manifest
        model_dir = properties.get("model_dir")
        self.model = XGBClassifier()
        if "serializedFile" in self.manifest["model"]:
            serialized_file = self.manifest["model"]["serializedFile"]
            model_weights = os.path.join(model_dir, serialized_file)
            self.model.load_model(model_weights)

        mapping_file_path = os.path.join(model_dir, "index_to_name.json")
        self.mapping = load_label_mapping(mapping_file_path)

        logger.info(
            f"XGBoost Classifier for iris dataset with weights {model_weights} loaded successfully"
        )
        self.initialized = True

    def preprocess(self, requests):
        inputs = []
        for row in requests:
            input = row.get("data") or row.get("body")
            if isinstance(input, (bytes, bytearray)):
                input = [float(value) for value in input.decode("utf-8").split(",")]
            inputs.append(input)
        return np.array(inputs)

    def inference(self, data):
        return self.model.predict(data)

    def postprocess(self, result):
        output = [self.mapping[str(res)] for res in result.tolist()]
        return output
