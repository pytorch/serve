""" The torchserve side inference end-points request are handled to
    return a KServe side response """
import logging
import pathlib
from typing import Dict, Union

import kserve
import orjson
from httpx import HTTPStatusError
from kserve.errors import ModelMissingError
from kserve.model import Model as Model
from kserve.protocol.infer_type import InferRequest

logging.basicConfig(level=kserve.constants.KSERVE_LOGLEVEL)

PREDICTOR_URL_FORMAT = "http://{0}/predictions/{1}"
EXPLAINER_URL_FORMAT = "http://{0}/explanations/{1}"
REGISTER_URL_FORMAT = "{0}/models?initial_workers=1&url={1}"
UNREGISTER_URL_FORMAT = "{0}/models/{1}"


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

        logging.info("Predict URL set to %s", self.predictor_host)
        self.explainer_host = self.predictor_host
        logging.info("Explain URL set to %s", self.explainer_host)

    async def _http_predict(
        self, payload: Union[Dict, InferRequest], headers: Dict[str, str] = None
    ) -> Dict:
        predict_url = PREDICTOR_URL_FORMAT.format(self.predictor_host, self.name)

        # Adjusting headers. Inject content type if not exist.
        # Also, removing host, as the header is the one passed to transformer and contains transformer's host
        predict_headers = {"Content-Type": "application/json"}
        if headers is not None:
            if "x-request-id" in headers:
                predict_headers["x-request-id"] = headers["x-request-id"]
            if "x-b3-traceid" in headers:
                predict_headers["x-b3-traceid"] = headers["x-b3-traceid"]
        if isinstance(payload, InferRequest):
            payload = payload.to_rest()
        data = orjson.dumps(payload)
        response = await self._http_client.post(
            predict_url, timeout=self.timeout, headers=predict_headers, content=data
        )
        if not response.is_success:
            message = (
                "{error_message}, '{0.status_code} {0.reason_phrase}' for url '{0.url}'"
            )
            error_message = ""
            if (
                "content-type" in response.headers
                and response.headers["content-type"] == "application/json"
            ):
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
