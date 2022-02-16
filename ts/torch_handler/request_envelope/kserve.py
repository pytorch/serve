"""
The KServe Envelope is used to handle the KServe
Input Request inside Torchserve.
"""
import json
import logging
from .base import BaseEnvelope

logger = logging.getLogger(__name__)


class KServeEnvelope(BaseEnvelope):
    """
    This function is used to handle the input request specified in kserve
    format and converts it into a Torchserve readable format.

    Args:
        data - List of Input Request in kserve Format
    Returns:
        [list]: Returns the list of the Input Request in Torchserve Format
    """

    def parse_input(self, data):
        self._data_list = [row.get("data") or row.get("body") for row in data]
        # selecting the first input from the list torchserve creates
        logger.debug("Parse input data_list %s", self._data_list)
        data = self._data_list[0]

        # If the KF Transformer and Explainer sends in data as bytesarray
        if isinstance(data, (bytes, bytearray)):

            data = data.decode()
            data = json.loads(data)
            logger.debug("Bytes array is %s", data)

        self._inputs = data.get("instances")
        logger.debug("kserve parsed inputs %s", self._inputs)
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
        logger.debug("The Response of kserve %s", data)
        if not self._is_explain():
            response["predictions"] = data
        else:
            response["explanations"] = data
        return [response]

    def _is_explain(self):
        if self.context and self.context.get_request_header(0, "explain"):
            if self.context.get_request_header(0, "explain") == "True":
                return True

        return False
