import json
from itertools import chain
from base64 import b64decode

from .base import BaseEnvelope

class KFservingEnvelope(BaseEnvelope):
    """
    Implementation. Captures batches in JSON format, returns
    also in JSON format.
    """
    _lengths = []
    _inputs = []
    _outputs = []
    _data_list = []

    def parse_input(self, data):
        print("Parsing input in KFServing.py")
        self._data_list = [row.get("data") or row.get("body") for row in data]
        self._inputs = [data.get("inputs") for data in self._data_list]
        print("KFServing parsed inputs", self._inputs)
        return self._inputs

    def format_output(self, data):
        self._outputs = [data.get("outputs") for data in self._data_list]
        self._outputs = self._outputs[0]
        response = {}
        for output in self._outputs:
            if isinstance(output, dict):
                response[output["name"]] = data[output["name"]]
        print("The output of KFServing", response)
        return [response]