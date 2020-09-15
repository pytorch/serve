import kfserving
import logging
logging.basicConfig(level=kfserving.constants.KFSERVING_LOGLEVEL)

class TorchserveModel(kfserving.KFModel):
    def __init__(self, name):
        super().__init__(name)
        
        logging.info("Predict URL set to %s", self.predictor_host)
        self.explainer_host = self.predictor_host
        logging.info("Explain URL set to %s", self.explainer_host)
        