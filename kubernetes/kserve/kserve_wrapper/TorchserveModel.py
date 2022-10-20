""" The torchserve side inference end-points request are handled to
    return a KServe side response """
import json
import logging
import pathlib
from importlib.metadata import version
from typing import Dict

import kserve
import tornado.web
from kserve.model import ModelMissingError

if version("kserve") >= "0.8.0":
    from kserve.model import Model as Model
else:
    from kserve.kfmodel import KFModel as Model

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

    async def predict(self, request: Dict) -> Dict:
        """The predict method is called when we hit the inference endpoint and handles
        the inference request and response from the Torchserve side and passes it on
        to the KServe side.

        Args:
            request (Dict): Input request from the http client side.

        Raises:
            NotImplementedError: If the predictor host on the KServe side is not
                                 available.

            tornado.web.HTTPError: If there is a bad response from the http client.

        Returns:
            Dict: The Response from the input from the inference endpoint.
        """
        if not self.predictor_host:
            raise NotImplementedError
        logging.debug("kfmodel predict request is %s", json.dumps(request))
        logging.info("PREDICTOR_HOST : %s", self.predictor_host)
        headers = {"Content-Type": "application/json; charset=UTF-8"}
        response = await self._http_client.fetch(
            PREDICTOR_URL_FORMAT.format(self.predictor_host, self.name),
            method="POST",
            request_timeout=self.timeout,
            headers=headers,
            body=json.dumps(request),
        )

        if response.code != 200:
            raise tornado.web.HTTPError(status_code=response.code, reason=response.body)
        return json.loads(response.body)

    async def explain(self, request: Dict) -> Dict:
        """The predict method is called when we hit the explain endpoint and handles the
        explain request and response from the Torchserve side and passes it on to the
        KServe side.

        Args:
            request (Dict): Input request from the http client side.

        Raises:
            NotImplementedError: If the predictor host on the KServe side is not
                                 available.

            tornado.web.HTTPError: If there is a bad response from the http client.

        Returns:
            Dict: The Response from the input from the explain endpoint.
        """
        if self.explainer_host is None:
            raise NotImplementedError
        logging.info("kfmodel explain request is %s", json.dumps(request))
        logging.info("EXPLAINER_HOST : %s", self.explainer_host)
        headers = {"Content-Type": "application/json; charset=UTF-8"}
        response = await self._http_client.fetch(
            EXPLAINER_URL_FORMAT.format(self.explainer_host, self.name),
            method="POST",
            request_timeout=self.timeout,
            headers=headers,
            body=json.dumps(request),
        )
        if response.code != 200:
            raise tornado.web.HTTPError(status_code=response.code, reason=response.body)
        return json.loads(response.body)

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
