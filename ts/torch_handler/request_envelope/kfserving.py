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
        #selecting the first input from the list torchserve creates
        data = self._data_list[0]
        self._inputs = data.get("instances") 
        #selecting the first input from the kfserving request instances list
        #self._inputs = self._inputs[0]
        print("KFServing parsed inputs", self._inputs)
        return self._inputs

    def format_output(self, output, output_explain):
        response = {}
        response["predictions"] = output
        if output_explain != None:
            response["explanations"] =  output_explain
  
        print("The Response of KFServing", response)
        return [response]