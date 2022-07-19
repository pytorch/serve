"""
Handler for Torchrec DLRM based recommendation system
"""
import json
import logging
import os
from abc import ABC

import torch
from torchrec.datasets.criteo import DEFAULT_CAT_NAMES
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

from ts.torch_handler.base_handler import BaseHandler, ipex_enabled

logger = logging.getLogger(__name__)


class TorchRecDLRMHandler(BaseHandler, ABC):
    """ """

    def initialize(self, context):
        """Initialize function loads the model.pt file and initialized the model object.
           This version creates and initialized the model on cpu fist and transfers to gpu in a second step to prevent GPU OOM.

        Args:
            context (context): It is a JSON Object containing information
            pertaining to the model artifacts parameters.

        Raises:
            RuntimeError: Raises the Runtime error when the model.py is missing

        """
        properties = context.system_properties

        # Set device to cpu to prevent GPU OOM errors
        self.device = "cpu"
        self.manifest = context.manifest

        model_dir = properties.get("model_dir")
        model_pt_path = None
        if "serializedFile" in self.manifest["model"]:
            serialized_file = self.manifest["model"]["serializedFile"]
            model_pt_path = os.path.join(model_dir, serialized_file)

        # model def file
        model_file = self.manifest["model"].get("modelFile", "")

        if not model_file:
            raise RuntimeError("model.py not specified")

        logger.debug("Loading eager model")
        self.model = self._load_pickled_model(model_dir, model_file, model_pt_path)

        self.map_location = (
            "cuda"
            if torch.cuda.is_available() and properties.get("gpu_id") is not None
            else "cpu"
        )

        self.device = torch.device(
            self.map_location + ":" + str(properties.get("gpu_id"))
            if torch.cuda.is_available() and properties.get("gpu_id") is not None
            else self.map_location
        )

        self.model.to(self.device)

        self.model.eval()
        if ipex_enabled:
            self.model = self.model.to(memory_format=torch.channels_last)
            self.model = ipex.optimize(self.model)

        logger.debug("Model file %s loaded successfully", model_pt_path)

        self.initialized = True

    def preprocess(self, data):
        """
        Description

        Args:
            data (str): The input data is in the form of a string

        Returns:
            (Tensor): KeyJaggedTensor is returned
            (str): The raw input is also returned in this function
        """

        line = data[0]
        json_data = line.get("data") or line.get("body")

        batch = json.loads(json_data)

        assert "float_features" in batch
        assert "id_list_features.lengths" in batch
        assert "id_list_features.values" in batch

        dense_features = torch.FloatTensor(batch["float_features"])

        sparse_features = KeyedJaggedTensor(
            keys=DEFAULT_CAT_NAMES,
            lengths=torch.LongTensor(batch["id_list_features.lengths"]),
            values=torch.FloatTensor(batch["id_list_features.values"]),
        )

        return dense_features, sparse_features

    def inference(self, data, *args, **kwargs):
        """
        Description

        Args:
            data (torch tensor): The data is in the form of Torch Tensor
                                 whose shape should match that of the
                                  Model Input shape.

        Returns:
            (Torch Tensor): The predicted response from the model is returned
                            in this function.
        """
        with torch.no_grad():
            data = map(lambda x: x.to(self.device), data)
            results = self.model(*data)
        return results

    def postprocess(self, data):
        """
        The post process function converts the prediction response into a
           Torchserve compatible format

        Args:
            data (Torch Tensor): The data parameter comes from the prediction output
            output_explain (None): Defaults to None.

        Returns:
            (list): Returns the response containing the predictions and explanations
                    (if the Endpoint is hit).It takes the form of a list of dictionary.
        """
        res = {"default": data.squeeze().float().tolist()}
        if data.shape[0] == 1:
            res["default"] = [res["default"]]

        return json.dumps(res)
