""" The repository to serve the Torchserve Models in the KFServing side"""
import logging
import kserve
from kserve.kfmodel_repository import KFModelRepository

logging.basicConfig(level=kserve.constants.KSERVE_LOGLEVEL)


class TSModelRepository(KFModelRepository):
    """A repository of KFServing KFModels
    Args:
        KFModelRepository (object): The parameters from the KFModelRepository is passed
        as inputs to the TSModel Repository.
    """

    def __init__(self, inference_address: str, management_address: str, model_dir: str):
        """The Inference Address, Management Address and the Model Directory from the KFServing
        side is initialized here.

        Args:
            inference_address (str): The Inference Address present in the KFServing side.
            management_address (str): The Management Address present in the kfserving side.
            model_dir (str): the directory of the model artefacts in the kfserving side.
        """
        super().__init__(model_dir)
        logging.info("TSModelRepo is initialized")
        self.inference_address = inference_address
        self.management_address = management_address
        self.model_dir = model_dir
