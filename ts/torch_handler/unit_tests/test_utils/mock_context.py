"""
Mocks for adding model context without loading all of Torchserve
"""

import uuid

import torch

from ts.metrics.metrics_store import MetricsStore


class MockContext:
    """
    Mock class to replicate the context passed into model initialize
    """

    def __init__(
        self,
        model_pt_file="model.pt",
        model_dir="ts/torch_handler/unit_tests/models/tmp",
        model_file="model.py",
        gpu_id="0",
        model_name="mnist",
    ):
        self.manifest = {"model": {}}
        if model_pt_file:
            self.manifest["model"]["serializedFile"] = model_pt_file

        if model_file:
            self.manifest["model"]["modelFile"] = model_file

        self.system_properties = {"model_dir": model_dir}

        if torch.cuda.is_available() and gpu_id:
            self.system_properties["gpu_id"] = gpu_id

        self.explain = False
        self.metrics = MetricsStore(uuid.uuid4(), model_name)

    def get_request_header(self, idx, exp):
        if idx and exp:
            if self.explain:
                return True
        return False
