""" A respository of image transformer models """
import logging

import kserve
from kserve.model_repository import ModelRepository as ModelRepository

logging.basicConfig(level=kserve.constants.KSERVE_LOGLEVEL)

logging.basicConfig(level=kserve.constants.KSERVE_LOGLEVEL)


class TransformerModelRepository(ModelRepository):
    """Stores the Image Transformer Models

    Args:
        KFModelRepository (class): KFModel Repository class object of
        kserve is passed here.
    """

    def __init__(self, predictor_host: str):
        """Initialize the Transformer Model Repository class object

        Args:
            predictor_host (str): The predictor host is specified here
        """
        super().__init__()
        logging.info("ImageTSModelRepo is initialized")
        self.predictor_host = predictor_host
