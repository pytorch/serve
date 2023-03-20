# pylint: disable=W0223
# Details : https://github.com/PyCQA/pylint/issues/3098
"""
Base module for all vision handlers
"""
import json
import os

import numpy as np
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy

from ts.torch_handler.image_classifier import ImageClassifier


class DALIHandler(ImageClassifier):
    """
    Base class for all vision handlers
    """

    def __init__(self):
        super(DALIHandler, self).__init__()

    def initialize(self, context):
        super().initialize(context)
        properties = context.system_properties
        self.model_dir = properties.get("model_dir")

        self.dali_file = [
            file for file in os.listdir(self.model_dir) if file.endswith(".dali")
        ]
        if not len(self.dali_file):
            raise RuntimeError("Missing dali pipeline file.")
        self.PREFETCH_QUEUE_DEPTH = 2
        dali_config_file = os.path.join(self.model_dir, "dali_config.json")
        if not os.path.isfile(dali_config_file):
            raise RuntimeError("Missing dali_config.json file.")
        with open(dali_config_file) as setup_config_file:
            self.dali_configs = json.load(setup_config_file)
        filename = os.path.join(self.model_dir, self.dali_file[0])
        self.pipe = Pipeline.deserialize(filename=filename)
        # pylint: disable=protected-access
        self.pipe._max_batch_size = self.dali_configs["batch_size"]
        self.pipe._num_threads = self.dali_configs["num_threads"]
        self.pipe._device_id = self.dali_configs["device_id"]

    def preprocess(self, data):
        """The preprocess function of MNIST program converts the input data to a float tensor

        Args:
            data (List): Input data from the request is in the form of a Tensor

        Returns:
            list : The preprocess function returns the input image as a list of float tensors.
        """
        batch_tensor = []

        input_byte_arrays = [i["body"] if "body" in i else i["data"] for i in data]
        for byte_array in input_byte_arrays:
            np_image = np.frombuffer(byte_array, dtype=np.uint8)
            batch_tensor.append(np_image)  # we can use numpy

        for _ in range(self.PREFETCH_QUEUE_DEPTH):
            self.pipe.feed_input("my_source", batch_tensor)

        datam = DALIGenericIterator(
            [self.pipe],
            ["data"],
            last_batch_policy=LastBatchPolicy.PARTIAL,
            last_batch_padded=True,
        )
        result = []
        for _, data in enumerate(datam):
            result.append(data[0]["data"])
            break

        return result[0].to(self.device)
