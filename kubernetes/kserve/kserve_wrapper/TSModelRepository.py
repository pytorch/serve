""" The repository to serve the Torchserve Models in the kserve side"""
import logging
from importlib.metadata import version

import kserve

if version("kserve") >= "0.8.0":
    from kserve.model_repository import ModelRepository as ModelRepository
else:
    from kserve.kfmodel_repository import KFModelRepository as ModelRepository

logging.basicConfig(level=kserve.constants.KSERVE_LOGLEVEL)


class TSModelRepository(ModelRepository):
    """A repository of kserve KFModels
    Args:
        KFModelRepository (object): The parameters from the KFModelRepository is passed
        as inputs to the TSModel Repository.
    """

    def __init__(self, inference_address: str, management_address: str, model_dir: str):
        """The Inference Address, Management Address and the Model Directory from the kserve
        side is initialized here.

        Args:
            inference_address (str): The Inference Address present in the kserve side.
            management_address (str): The Management Address present in the kserve side.
            model_dir (str): the directory of the model artefacts in the kserve side.
        """
        super().__init__(model_dir)
        logging.info("TSModelRepo is initialized")
        self.inference_address = inference_address
        self.management_address = management_address
        self.model_dir = model_dir
