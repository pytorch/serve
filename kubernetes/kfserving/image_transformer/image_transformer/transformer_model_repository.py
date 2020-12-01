""" A respository of image transformer models """
import logging
from kfserving.kfmodel_repository import KFModelRepository
import kfserving

logging.basicConfig(level=kfserving.constants.KFSERVING_LOGLEVEL)



class TransformerModelRepository(KFModelRepository):
    """Stores the Image Transformer Models

    Args:
        KFModelRepository (class): KFModel Repository class object of
        kfserving is passed here.
    """

    def __init__(self, predictor_host: str):
        """Initialize the Transformer Model Repository class object

        Args:
            predictor_host (str): The predictor host is specified here
        """
        super().__init__()
        logging.info("ImageTSModelRepo is initialized")
        self.predictor_host = predictor_host
