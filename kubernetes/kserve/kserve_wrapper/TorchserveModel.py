""" The torchserve side inference end-points request are handled to
    return a KServe side response """
import logging
import pathlib
from enum import Enum
from typing import Dict, Union

import grpc
import kserve
from gprc_utils import to_ts_grpc
from inference_pb2_grpc import InferenceAPIsServiceStub
from kserve.errors import ModelMissingError
from kserve.model import Model as Model
from kserve.protocol.grpc.grpc_predict_v2_pb2 import (
    ModelInferRequest,
    ModelInferResponse,
)
from kserve.protocol.infer_type import InferRequest

logging.basicConfig(level=kserve.constants.KSERVE_LOGLEVEL)

PREDICTOR_URL_FORMAT = PREDICTOR_V2_URL_FORMAT = "http://{0}/predictions/{1}"
EXPLAINER_URL_FORMAT = EXPLAINER_v2_URL_FORMAT = "http://{0}/explanations/{1}"
REGISTER_URL_FORMAT = "{0}/models?initial_workers=1&url={1}"
UNREGISTER_URL_FORMAT = "{0}/models/{1}"


class PredictorProtocol(Enum):
    REST_V1 = "v1"
    REST_V2 = "v2"
    GRPC_V2 = "grpc-v2"


PREDICTOR_URL_FORMAT = "http://{0}/v1/models/{1}:predict"
EXPLAINER_URL_FORMAT = "http://{0}/v1/models/{1}:explain"


class TorchserveModel(Model):
    """The torchserve side inference and explain end-points requests are handled to
    return a KServe side response

    Args:
        kserve.KFModel(class object): The predict and explain methods are overridden by torchserve
        side predict and explain http requests.
    """

    def __init__(
        self,
        name,
        inference_address,
        management_address,
        grpc_inference_address,
        protocol,
        model_dir,
    ):
        """The Model Name, Inference Address, Management Address and the model directory
        are specified.

        Args:
            name (str): Model Name
            inference_address (str): The Inference Address in which we hit the inference end point
            management_address (str): The Management Address in which we register the model.
            model_dir (str): The location of the model artifacts.
        """
        super().__init__(name)

        if not self.predictor_host:
            self.predictor_host = inference_address.split("//")[1]
        if not self.explainer_host:
            self.explainer_host = self.predictor_host

        self.inference_address = inference_address
        self.management_address = management_address
        self.grpc_inference_address = grpc_inference_address
        self.model_dir = model_dir
        self.protocol = protocol

        if self._grpc_client_stub == None:
            self._channel = grpc.aio.insecure_channel(self.grpc_inference_address)
            self._grpc_client_stub = InferenceAPIsServiceStub(self._channel)

        logging.info("Predict URL set to %s", self.predictor_host)
        self.explainer_host = self.predictor_host
        logging.info("Explain URL set to %s", self.explainer_host)

    async def _grpc_predict(
        self,
        payload: Union[ModelInferRequest, InferRequest],
        headers: Dict[str, str] = None,
    ) -> ModelInferResponse:
        if isinstance(payload, InferRequest):
            payload = to_ts_grpc(payload)
        async_result = await self._grpc_client.Predictions(payload)
        return async_result

    def load(self) -> bool:
        """This method validates model availabilty in the model directory
        and sets ready flag to true.
        """
        model_path = pathlib.Path(kserve.Storage.download(self.model_dir))
        paths = list(pathlib.Path(model_path).glob("*.mar"))
        existing_paths = [path for path in paths if path.exists()]
        if len(existing_paths) == 0:
            raise ModelMissingError(model_path)
        self.ready = True
        return self.ready
