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
  """
  This function is used to handle the input request specified in KFServing
  format and converts it into a Torchserve readable format.

  Args:
      data - List of Input Request in KFServing Format
  Returns:
      [list]: Returns the list of the Input Request in Torchserve Format
  """

  def parse_input(self, data):
    logger.info("Parsing input in KFServingv2.py")
    self._data_list = [row.get("data") or row.get("body") for row in data]
    self._data_list = [row for row in self._data_list[0].get("inputs")]
    # selecting the first input from the list torchserve creates
    logger.info("Parse input data_list %s", self._data_list)
   
    # If the KF Transformer and Explainer sends in data as bytesarray
    # if isinstance(data, (bytes, bytearray)):

    #     data = data.decode()
    #     data = json.loads(data)
    #     logger.info("Bytes array is %s", data)

    self._inputs = self._data_list
    logger.info("KFServingv2 parsed inputs %s", self._inputs)
    return self._inputs

  def format_output(self, data):
    """
    Returns the prediction response and captum explanation response of the input request.

    Args:
        outputs (List): The outputs arguments is in the form of a list of dictionaries.

    Returns:
        (list): The response is returned as a list of predictions and explanations
    """

    response = {}
    output = {}

    logger.info("The Response of KFServingv2 %s", data)
    response["model_name"] = self.context.manifest.get("model").get("modelName")
    response["model_version"] = self.context.manifest.get("model").get("modelVersion")
    response["id"] = self.context.get_request_id(0)
    output["name"] = "explain" if self.context.get_request_header(0, "explain") == "True" else "predict"
    output["shape"] = np.array(data).shape
    output["datatype"] = _to_datatype(np.array(data).dtype)
    output["data"] = data
    response["outputs"] = [output]
    return [response]
