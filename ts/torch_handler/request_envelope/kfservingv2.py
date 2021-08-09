"""
The KFServing Envelope is used to handle the KFServing
Input Request inside Torchserve.
"""
import json
import numpy as np
import logging
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


def _to_dtype(datatype: str) -> "np.dtype":
    dtype = _DatatypeToNumpy[datatype]
    return np.dtype(dtype)


def _to_datatype(dtype: np.dtype) -> str:
    as_str = str(dtype)
    datatype = _NumpyToDatatype[as_str]

    return datatype

class KFservingv2Envelope(BaseEnvelope):
    def parse_input(self, data):
        logger.info("Parsing input in KFServingv2.py")
        inputs = self._batch_from_json(data)
    
        # If the KF Transformer and Explainer sends in data as bytesarray
        if isinstance(data, (bytes, bytearray)):

            data = data.decode()
            data = json.loads(data)
            logger.info("Bytes array is %s", data)

        logger.info("KFServingv2 parsed inputs %s", inputs)
        return inputs

    def _batch_from_json(self, rows):
        """
        Joins the instances of a batch of JSON objects
        """
        logger.info("Parse input data %s", rows)
        body_list = rows[0].get("body")
        data_list = self._from_json(body_list)
        return data_list

    def _from_json(self, body_list):
        """
        Extracts the data from the JSON object
        """
        data_list = [row for row in body_list.get("inputs")]
        return data_list

    def format_output(self, data):
        logger.info("The Response of KFServingv2 %s", data)
        response = {}
        response["id"] = self.context.get_request_id(0)
        response["model_name"] = self.context.manifest.get("model").get("modelName")
        response["model_version"] = self.context.manifest.get("model").get("modelVersion")
        response["outputs"] = self._batch_to_json(data)
        return [response]

    def _batch_to_json(self, data):
        """
        Splits batch output to json objects
        """
        output = []
        for item in data:
            output.append(self._to_json(item))
        return output

    def _to_json(self, data):
        """
        Constructs JSON object from data
        """
        output_data = {}
        data_ndarray = np.ndarray(data)
        output_data["name"] = "explain" if self.context.get_request_header(0, "explain") == "True" else "predict"
        output_data["shape"] = data_ndarray.shape
        output_data["datatype"] = _to_datatype(data_ndarray.dtype)
        output_data["data"] = data_ndarray.flatten().tolist()
        return output_data
