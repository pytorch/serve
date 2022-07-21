"""
Handler for Torchrec DLRM based recommendation system
"""
import json
import logging
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

        line = data[0]
        json_data = line.get("data") or line.get("body")

        batch = json.loads(json_data)

        # This is the dense feature part
        assert "float_features" in batch
        # The sparse input consists of a length vector and the values.
        # The length vector contains the number of elements which are part fo the sam entry in the linear list provided as input.
        assert "id_list_features.lengths" in batch
        assert "id_list_features.values" in batch

        dense_features = torch.FloatTensor(batch["float_features"])

        sparse_features = KeyedJaggedTensor(
            keys=DEFAULT_CAT_NAMES,
            lengths=torch.LongTensor(batch["id_list_features.lengths"]),
            values=torch.zeros_like(
                torch.FloatTensor(batch["id_list_features.values"])
            ),
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
            (list): Returns the response containing the predictions which consist of a single score per batch entry.
        """
        res = {"default": data.squeeze().float().tolist()}
        if data.shape[0] == 1:
            res["default"] = [res["default"]]

        return [res]
