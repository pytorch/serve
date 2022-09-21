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

from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)


class TorchRecDLRMHandler(BaseHandler, ABC):
    """
    Handler for TorchRec DLRM example
    """

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

        logger.debug("Model file %s loaded successfully", model_pt_path)

        self.initialized = True

    def preprocess(self, data):
        """
        The input values for the DLRM model are twofold. There is a dense part and a sparse part.
        The sparse part consists of a list of ids where each entry can consist of zero, one or multiple ids.
        Due to the inconsistency in elements, the sparse part is represented by the KeyJaggedTensor class provided by TorchRec.

        Args:
            data (str): The input data is in the form of a string

        Returns:
            Tuple of:
                (Tensor): Dense features
                (KeyJaggedTensor): Sparse features
        """

        float_features, id_list_features_lengths, id_list_features_values = [], [], []

        for row in data:

            input = row.get("data") or row.get("body")

            if not isinstance(input, dict):
                input = json.loads(input)

            # This is the dense feature part
            assert "float_features" in input
            # The sparse input consists of a length vector and the values.
            # The length vector contains the number of elements which are part fo the same entry in the linear list provided as input.
            assert "id_list_features.lengths" in input
            assert "id_list_features.values" in input

            float_features.append(input["float_features"])
            id_list_features_lengths.extend(input["id_list_features.lengths"])
            id_list_features_values.append(input["id_list_features.values"])

        # Reformat the values input for KeyedJaggedTensor
        id_list_features_values = torch.FloatTensor(id_list_features_values)
        id_list_features_values = torch.transpose(id_list_features_values, 0, 1)
        id_list_features_values = [value for value in id_list_features_values]

        # Dense and Sparse Features for DLRM model
        dense_features = torch.FloatTensor(float_features)
        sparse_features = KeyedJaggedTensor(
            keys=DEFAULT_CAT_NAMES,
            lengths=torch.LongTensor(id_list_features_lengths),
            values=torch.cat(id_list_features_values),
        )

        return dense_features, sparse_features

    def inference(self, data):
        """
        The inference call moves the elements of the tuple onto the device and calls the model

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
            (list): Returns the response containing the predictions which consist of a single score per input entry.
        """

        result = []
        for item in data:
            res = {}
            res["score"] = item.squeeze().float().tolist()
            result.append(res)

        return result
