# pylint: disable=W0223
# Details : https://github.com/PyCQA/pylint/issues/3098
"""
Base module for all vision handlers
"""
import json
import os

import numpy as np
import torch
from nvidia.dali.pipeline import Pipeline

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
        dali_config_file = os.path.join(self.model_dir, "dali_config.json")
        if not os.path.isfile(dali_config_file):
            raise RuntimeError("Missing dali_config.json file.")
        with open(dali_config_file) as setup_config_file:
            self.dali_configs = json.load(setup_config_file)
        dali_filename = os.path.join(self.model_dir, self.dali_file[0])
        self.pipe = Pipeline.deserialize(
            filename=dali_filename,
            batch_size=self.dali_configs["batch_size"],
            num_threads=self.dali_configs["num_threads"],
            prefetch_queue_depth=1,
            device_id=self.dali_configs["device_id"],
            seed=self.dali_configs["seed"],
        )
        self.pipe.build()
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
        result = []

        input_byte_arrays = [i["body"] if "body" in i else i["data"] for i in data]
        for byte_array in input_byte_arrays:
            np_image = np.frombuffer(byte_array, dtype=np.uint8)
            batch_tensor.append(np_image)  # we can use numpy

        response = self.pipe.run(source=batch_tensor)
        for idx, _ in enumerate(response[0]):
            data = torch.tensor(response[0].at(idx))
            result.append(data.unsqueeze(0))

        return torch.cat(result).to(self.device)
