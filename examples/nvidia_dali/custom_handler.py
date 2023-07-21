# pylint: disable=W0223
# Details : https://github.com/PyCQA/pylint/issues/3098
"""
Base module for all vision handlers
"""
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

        if "dali" in context.model_yaml_config:
            self.batch_size = context.model_yaml_config["dali"]["batch_size"]
            self.num_threads = context.model_yaml_config["dali"]["num_threads"]
            self.device_id = context.model_yaml_config["dali"]["device_id"]
            if "pipeline_file" in context.model_yaml_config["dali"]:
                pipeline_filename = context.model_yaml_config["dali"]["pipeline_file"]
                pipeline_filepath = os.path.join(self.model_dir, pipeline_filename)
            else:
                raise RuntimeError("Missing dali pipeline file.")
            if not os.path.exists(pipeline_filepath):
                raise RuntimeError("Dali pipeline file not found!")
            self.pipeline = Pipeline.deserialize(
                filename=pipeline_filepath,
                batch_size=self.batch_size,
                num_threads=self.num_threads,
                prefetch_queue_depth=1,
                device_id=self.device_id,
                seed=self.seed,
            )
            # pylint: disable=protected-access
            self.pipeline._max_batch_size = self.batch_size
            self.pipeline._num_threads = self.num_threads
            self.pipeline._device_id = self.device_id

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
