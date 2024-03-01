""" The torchserve side inference end-points request are handled to
    return a KServe side response """
import logging
import os
import pathlib
import time
from enum import Enum
from typing import Dict, Union

import grpc
import inference_pb2_grpc
import kserve
import requests
from gprc_utils import from_ts_grpc, to_ts_grpc
from inference_pb2 import PredictionResponse
from kserve.errors import ModelMissingError
from kserve.model import Model as Model
from kserve.protocol.grpc.grpc_predict_v2_pb2 import (
    ModelInferRequest,
    ModelInferResponse,
)
from kserve.protocol.infer_type import InferRequest, InferResponse
from kserve.storage import Storage

logging.basicConfig(level=kserve.constants.KSERVE_LOGLEVEL)

PREDICTOR_URL_FORMAT = PREDICTOR_V2_URL_FORMAT = "http://{0}/predictions/{1}"
EXPLAINER_URL_FORMAT = EXPLAINER_v2_URL_FORMAT = "http://{0}/explanations/{1}"
REGISTER_URL_FORMAT = "{0}/models?initial_workers=1&url={1}"
UNREGISTER_URL_FORMAT = "{0}/models/{1}"
READINESS_URL_FORMAT = "{0}/models/{1}?customized={2}"


class PredictorProtocol(Enum):
    REST_V1 = "v1"
    REST_V2 = "v2"
    GRPC_V2 = "grpc-v2"


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
        self.model_dir = model_dir

        # Validate the protocol value passed otherwise, the default value will be used
        if protocol is not None:
            self.protocol = PredictorProtocol(protocol).value

        if self.protocol == PredictorProtocol.GRPC_V2.value:
            self.predictor_host = grpc_inference_address

        logging.info("Predict URL set to %s", self.predictor_host)
        logging.info("Explain URL set to %s", self.explainer_host)
        logging.info("Protocol version is %s", self.protocol)

    def grpc_client(self):
        if self._grpc_client_stub is None:
            self.channel = grpc.aio.insecure_channel(self.predictor_host)
            self.grpc_client_stub = inference_pb2_grpc.InferenceAPIsServiceStub(
                self.channel
            )
        return self.grpc_client_stub

    async def _grpc_predict(
        self,
        payload: Union[ModelInferRequest, InferRequest],
        headers: Dict[str, str] = None,
    ) -> ModelInferResponse:
        """Overrides the `_grpc_predict` method in Model class. The predict method calls
        the `_grpc_predict` method if the self.protocol is "grpc_v2"

        Args:
            request (Dict|InferRequest|ModelInferRequest): The response passed from ``predict`` handler.

        Returns:
            Dict: Torchserve grpc response.
        """
        payload = to_ts_grpc(payload)
        grpc_stub = self.grpc_client()
        async_result = await grpc_stub.Predictions(payload)
        return from_ts_grpc(async_result)

    def postprocess(
        self,
        response: Union[Dict, InferResponse, ModelInferResponse, PredictionResponse],
        headers: Dict[str, str] = None,
    ) -> Union[Dict, ModelInferResponse]:
        """This method converts the v2 infer response types to gRPC or REST.
        For gRPC request it converts InferResponse to gRPC message or directly returns ModelInferResponse from
        predictor call or converts TS PredictionResponse to ModelInferResponse.
        For REST request it converts ModelInferResponse to Dict or directly returns from predictor call.

        Args:
            response (Dict|InferResponse|ModelInferResponse|PredictionResponse): The response passed from ``predict`` handler.
            headers (Dict): Request headers.

        Returns:
            Dict: post-processed response.
        """
        if headers:
            if "grpc" in headers.get("user-agent", ""):
                if isinstance(response, ModelInferResponse):
                    return response
                elif isinstance(response, InferResponse):
                    return response.to_grpc()
                elif isinstance(response, PredictionResponse):
                    return from_ts_grpc(response)
            if "application/json" in headers.get("content-type", ""):
                # If the original request is REST, convert the gRPC predict response to dict
                if isinstance(response, ModelInferResponse):
                    return InferResponse.from_grpc(response).to_rest()
                elif isinstance(response, InferResponse):
                    return response.to_rest()
        return response

    def load(self) -> bool:
        """This method validates model availabilty in the model directory
        and sets ready flag to true.
        """
        model_path = pathlib.Path(Storage.download(self.model_dir))
        paths = list(pathlib.Path(model_path).glob("*"))
        existing_paths = [
            path for path in paths if path.is_dir() or path.suffixes == [".mar"]
        ]
        if len(existing_paths) == 0:
            raise ModelMissingError(model_path)

        num_try = 0
        model_load_customized = os.environ.get("MODEL_LOAD_CUSTOMIZED", "false")
        model_load_max_try = int(os.environ.get("MODEL_LOAD_MAX_TRY", 10))
        model_load_delay = int(os.environ.get("MODEL_LOAD_DELAY", 30))
        model_load_timeout = int(os.environ.get("MODEL_LOAD_TIMEOUT", 5))
        while num_try < model_load_max_try and not self.ready:
            num_try = num_try + 1
            logging.info(
                f"Loading {self.name} .. {num_try} of {model_load_max_try} tries.."
            )

            try:
                response = requests.get(
                    READINESS_URL_FORMAT.format(
                        self.management_address, self.name, model_load_customized
                    ),
                    timeout=model_load_timeout,
                ).json()

                default_verison = response[0]

                workers = default_verison["workers"]
                workers_status = [
                    worker["id"] for worker in workers if worker["status"] == "READY"
                ]

                worker_ready = False
                if len(workers_status) > 0:
                    worker_ready = True

                self.ready = (
                    worker_ready
                    if model_load_customized == "false"
                    else worker_ready and "customizedMetadata" in default_verison
                )

            except (
                requests.ConnectionError,
                requests.Timeout,
                requests.ConnectTimeout,
                requests.ReadTimeout,
            ) as e:
                logging.info(f"The model {self.name} is not ready")

            except Exception as e:
                logging.info(e)
                logging.info(f"Failed loading model {self.name}")
                break

            logging.info(f"Sleep {model_load_delay} seconds for load {self.name}..")
            time.sleep(model_load_delay)

        if self.ready:
            logging.info(f"The model {self.name} is ready")

        return self.ready
