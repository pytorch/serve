import logging
from abc import ABC
from typing import Dict

from lru import LRU

from ts.context import Context
from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)


class StatefulHandler(BaseHandler, ABC):
    DEFAULT_CAPACITY = 10

    def __init__(self):
        super().__init__()
        self.cache: LRU = None
        self.sequence_ids: Dict = None
        self.context = None

    def initialize(self, ctx: Context):
        """
        Loads the model and Initializes the necessary artifacts
        """

        super().initialize(ctx)
        self.context = ctx
        if self.context.model_yaml_config["handler"] is not None:
            try:
                self.cache = LRU(
                    int(self.context.model_yaml_config["handler"]["cache"]["capacity"])
                )
            except KeyError:
                logger.error("No cache capacity was set! Using default value.")
                self.cache = LRU(StatefulHandler.DEFAULT_CAPACITY)

        self.initialized = True

    def preprocess(self, data):
        """
        Preprocess function to convert the request input to a tensor(Torchserve supported format).
        The user needs to override to customize the pre-processing

        Args :
            data (list): List of the data from the request input.

        Returns:
            tensor: Returns the tensor data of the input
        """

        self.sequence_ids = {}
        results = []
        for idx, row in enumerate(data):
            sequence_id = self.context.get_sequence_id(idx)

            prev = int(0)
            if self.cache.has_key(sequence_id):
                prev = int(self.cache[sequence_id])

            request = row.get("data") or row.get("body")
            if isinstance(request, (bytes, bytearray)):
                request = request.decode("utf-8")

            val = prev + int(request)
            self.cache[sequence_id] = val
            results.append(val)

        return results

    def inference(self, data, *args, **kwargs):
        return data

    def postprocess(self, data):
        """
        The post process function makes use of the output from the inference and converts into a
        Torchserve supported response output.

        Returns:
            List: The post process function returns a list of the predicted output.
        """

        return data
