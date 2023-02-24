""" The torchserve side inference end-points request are handled to
    return a KServe side response """
import orjson
import logging
import pathlib
from typing import Dict

import kserve
import httpx
from httpx import HTTPStatusError
from kserve.model import Model as Model
from kserve.errors import ModelMissingError
from kserve.protocol.infer_type import InferRequest, InferResponse

logging.basicConfig(level=kserve.constants.KSERVE_LOGLEVEL)

REGISTER_URL_FORMAT = "{0}/models?initial_workers=1&url={1}"
UNREGISTER_URL_FORMAT = "{0}/models/{1}"

PREDICTOR_URL_FORMAT = "http://{0}/v1/models/{1}:predict"
EXPLAINER_URL_FORMAT = "http://{0}/v1/models/{1}:explain"


class TorchserveModel(Model):
    """The torchserve side inference and explain end-points requests are handled to
    return a KServe side response

    Args:
        kserve.KFModel(class object): The predict and explain methods are overridden by torchserve
        side predict and explain http requests.
    """

    def __init__(self, name, inference_address, management_address, model_dir):
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

        logging.info("kfmodel Predict URL set to %s", self.predictor_host)
        self.explainer_host = self.predictor_host
        logging.info("kfmodel Explain URL set to %s", self.explainer_host)

    async def predict(self, payload: Dict, headers: Dict) -> Dict:
        """The predict method is called when we hit the inference endpoint and handles
        the inference request and response from the Torchserve side and passes it on
        to the KServe side.

        Args:
            payload (Dict): Input payload from the http client side.
            headers (Dict): Request headers.

        Raises:
            NotImplementedError: If the predictor host on the KServe side is not
                                 available.

            HTTPStatusError: If there is a bad response from the http client.

        Returns:
            Dict: The Response from the input from the inference endpoint.
        """
        if not self.predictor_host:
            raise NotImplementedError
        if isinstance(payload, InferRequest):
            payload = payload.to_rest()
        data = orjson.dumps(payload)
        logging.debug("kfmodel predict request is %s", data)
        logging.info("PREDICTOR_HOST : %s", self.predictor_host)
        predict_headers = {'Content-Type': 'application/json; charset=UTF-8'}
        if headers is not None:
            if 'X-Request-Id' in headers:
                predict_headers['X-Request-Id'] = headers['X-Request-Id']
            if 'X-B3-Traceid' in headers:
                predict_headers['X-B3-Traceid'] = headers['X-B3-Traceid']
        response = await self._http_client.post(
            PREDICTOR_URL_FORMAT.format(self.predictor_host, self.name),
            timeout=self.timeout,
            headers=predict_headers,
            content=data
        )
        if not response.is_success:
            message = (
                "{error_message}, '{0.status_code} {0.reason_phrase}' for url '{0.url}'"
            )
            error_message = ""
            if "content-type" in response.headers and response.headers["content-type"] == "application/json":
                error_message = response.json()
                if "error" in error_message:
                    error_message = error_message["error"]
            message = message.format(response, error_message=error_message)
            raise HTTPStatusError(message, request=response.request, response=response)
        return orjson.loads(response.content)

    async def explain(self, payload: Dict, headers: Dict) -> Dict:
        """The predict method is called when we hit the explain endpoint and handles the
        explain request and response from the Torchserve side and passes it on to the
        KServe side.

        Args:
            payload (Dict): Input payload from the http client side.
            headers (Dict): Request headers.

        Raises:
            NotImplementedError: If the explainer host on the KServe side is not
                                 available.

            HTTPStatusError: If there is a bad response from the http client.

        Returns:
            Dict: The Response from the input from the explain endpoint.
        """
        if self.explainer_host is None:
            raise NotImplementedError
        if isinstance(payload, InferRequest):
            payload = payload.to_rest()
        data = orjson.dumps(payload)
        logging.info("kfmodel explain request is %s", data)
        logging.info("EXPLAINER_HOST : %s", self.explainer_host)
        predict_headers = {"Content-Type": "application/json; charset=UTF-8"}
        if headers is not None:
            if 'X-Request-Id' in headers:
                predict_headers['X-Request-Id'] = headers['X-Request-Id']
            if 'X-B3-Traceid' in headers:
                predict_headers['X-B3-Traceid'] = headers['X-B3-Traceid']
        response = await self._http_client.post(
            EXPLAINER_URL_FORMAT.format(self.explainer_host, self.name),
            timeout=self.timeout,
            headers=predict_headers,
            content=data,
        )
        if not response.is_success:
            message = (
                "{error_message}, '{0.status_code} {0.reason_phrase}' for url '{0.url}'"
            )
            error_message = ""
            if "content-type" in response.headers and response.headers["content-type"] == "application/json":
                error_message = response.json()
                if "error" in error_message:
                    error_message = error_message["error"]
            message = message.format(response, error_message=error_message)
            raise HTTPStatusError(message, request=response.request, response=response)
        return orjson.loads(response.content)

    def load(self) -> bool:
        """This method validates model availabilty in the model directory
        and sets ready flag to true.
        """
        model_path = pathlib.Path(kserve.Storage.download(self.model_dir))
        paths = list(pathlib.Path(model_path).glob("*.mar"))
        existing_paths = [path for path in paths if path.exists()]
        if len(existing_paths) == 0:
            raise ModelMissingError(model_path)
        elif len(existing_paths) > 1:
            raise RuntimeError(
                "More than one model file is detected, "
                f"Only one is allowed within model_dir: {existing_paths}"
            )
        self.ready = True
        return self.ready
