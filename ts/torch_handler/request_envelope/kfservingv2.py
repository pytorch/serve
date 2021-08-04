"""
The KFServing Envelope is used to handle the KFServing
Input Request inside Torchserve.
"""
import json
import numpy as np
import logging
from .base import BaseEnvelope

logger = logging.getLogger(__name__)


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
    output["datatype"] = str(np.array(data).dtype)
    output["data"] = data
    response["outputs"] = [output]
    return [response]
