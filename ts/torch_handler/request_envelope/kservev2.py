"""
The KServe Envelope is used to handle the KServe
Input Request inside Torchserve.
"""
import json
import logging
from typing import Optional

import numpy as np

from .base import BaseEnvelope

logger = logging.getLogger(__name__)

_DatatypeToNumpy = {
    "BOOL": "bool",
    "UINT8": "uint8",
    "UINT16": "uint16",
    "UINT32": "uint32",
    "UINT64": "uint64",
    "INT8": "int8",
    "INT16": "int16",
    "INT32": "int32",
    "INT64": "int64",
    "FP16": "float16",
    "FP32": "float32",
    "FP64": "float64",
    "BYTES": "byte",
}

_NumpyToDatatype = {value: key for key, value in _DatatypeToNumpy.items()}

# NOTE: numpy has more types than v2 protocol
_NumpyToDatatype["object"] = "BYTES"

# Adding support for unicode string
# Ref: https://numpy.org/doc/stable/reference/arrays.dtypes.html
_NumpyToDatatype["U"] = "BYTES"


def _to_dtype(datatype: str) -> "np.dtype":
    dtype = _DatatypeToNumpy[datatype]
    return np.dtype(dtype)


def _to_datatype(dtype: np.dtype) -> str:
    as_str = str(dtype)
    if as_str not in _NumpyToDatatype:
        as_str = getattr(dtype, "kind")
    datatype = _NumpyToDatatype[as_str]

    return datatype


class KServev2Envelope(BaseEnvelope):
    """Implementation. Captures batches in KServe v2 protocol format, returns
    also in FServing v2 protocol format.
    """

    def parse_input(self, data):
        """Translates KServe request input to list of data expected by Torchserve.

        Parameters:
        data (json): KServe v2 request input json.
        {
          "inputs": [{
            "name": "input-0",
            "shape": [37],
            "datatype": "INT64",
            "data": [66, 108, 111, 111, 109]
          }]
        }

        Returns: list of data objects.
        [{
        'name': 'input-0',
        'shape': [5],
        'datatype': 'INT64',
        'data': [66, 108, 111, 111, 109]
        }]

        """
        logger.debug("Parsing input in KServe v2 format %s", data)
        inputs = self._batch_from_json(data)
        logger.debug("KServev2 parsed inputs %s", inputs)
        return inputs

    def _batch_from_json(self, rows):
        """
        Joins the instances of a batch of JSON objects
        """
        logger.debug("Parse input data %s", rows)
        body_list = [
            body_list.get("data") or body_list.get("body") for body_list in rows
        ]
        data_list = self._from_json(body_list)
        return data_list

    def _from_json(self, body_list):
        """
        Extracts the data from the JSON object
        """
        if isinstance(body_list[0], (bytes, bytearray)):
            body_list = [json.loads(body.decode("utf8")) for body in body_list]
            logger.debug("Bytes array is %s", body_list)

        input_names = []
        parameters = []
        ids = []
        input_parameters = []
        data_list = []

        for body in body_list:
            id = body.get("id")
            ids.append(id)
            params = body.get("parameters")
            if params:
                parameters.append(params)
            inp_names = []
            inp_params = []
            for i, input in enumerate(body["inputs"]):
                params = input.get("parameters")
                if params:
                    inp_params.append(params)
                if input["datatype"] == "BYTES":
                    body["inputs"][i]["data"] = input["data"][0]
                else:
                    body["inputs"][i]["data"] = (
                        np.array(input["data"]).reshape(tuple(input["shape"])).tolist()
                    )
                inp_names.append(input["name"])
            data = body["inputs"] if len(body["inputs"]) > 1 else body["inputs"][0]
            data_list.append(data)

            input_parameters.append(inp_params)
            input_names.append(inp_names)

        setattr(self.context, "input_request_id", ids)
        setattr(self.context, "input_names", input_names)
        setattr(self.context, "request_parameters", parameters)
        setattr(self.context, "input_parameters", input_parameters)
        logger.debug("Data array is %s", data_list)
        logger.debug("Request paraemeters array is %s", parameters)
        logger.debug("Input parameters is %s", input_parameters)
        return data_list

    def format_output(self, data):
        """Translates Torchserve output KServe v2 response format.

        Parameters:
        data (list): Torchserve response for handler.

        Returns: KServe v2 response json.
        {
          "id": "f0222600-353f-47df-8d9d-c96d96fa894e",
          "model_name": "bert",
          "model_version": "1",
          "outputs": [{
            "name": "input-0",
            "shape": [1],
            "datatype": "INT64",
            "data": [2]
          }]
        }

        """
        logger.debug("The Response of KServe v2 format %s", data)
        return self._batch_to_json(data)

    def _batch_to_json(self, batch: dict):
        """
        Splits batch output to json objects
        """
        parameters = getattr(self.context, "request_parameters")
        ids = getattr(self.context, "input_request_id")
        input_parameters = getattr(self.context, "input_parameters")
        responses = []
        for index, data in enumerate(batch):
            response = {}
            response["id"] = ids[index] or self.context.get_request_id(index)
            if parameters and parameters[index]:
                response["parameters"] = parameters[index]
            response["model_name"] = self.context.manifest.get("model").get("modelName")
            response["model_version"] = self.context.manifest.get("model").get(
                "modelVersion"
            )
            outputs = []
            if isinstance(data, dict):
                for key, item in data.items():
                    outputs.append(self._to_json(item, key, input_parameters))
            else:
                outputs.append(self._to_json(data, "predictions", input_parameters))
            response["outputs"] = outputs
            responses.append(response)
        delattr(self.context, "input_names")
        delattr(self.context, "input_request_id")
        delattr(self.context, "input_parameters")
        delattr(self.context, "request_parameters")
        return responses

    def _to_json(self, data, output_name, parameters: Optional[list] = None):
        """
        Constructs JSON object from data
        """
        output_data = {}
        data_ndarray = np.array(data).flatten()
        output_data["name"] = output_name
        if parameters:
            output_data["parameters"] = parameters
        output_data["datatype"] = _to_datatype(data_ndarray.dtype)
        output_data["data"] = data_ndarray.tolist()
        output_data["shape"] = data_ndarray.flatten().shape
        return output_data
