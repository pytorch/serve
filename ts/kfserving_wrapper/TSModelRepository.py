import os
from kfserving.kfmodel_repository import KFModelRepository, MODEL_MOUNT_DIRS
from TorchserveModel import TorchserveModel


class TSModelRepository(KFModelRepository):

    def __init__(self, inference_address: str, management_address: str, model_dir: str):
        super().__init__(model_dir)
        print("TSModelRepo is initialized")
        self.inference_address = inference_address
        self.management_address = management_address
        self.model_dir = model_dir

        