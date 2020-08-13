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

    def format_output(self, results):
        self._outputs = [data_2.get("outputs") for data_2 in self._data_list]
        #Processing only the first output, as we are not handling batch inference
        self._outputs = self._outputs[0]
        output_dict = {}
        outputs_list = []
        
        #Processing only the first output, as we are not handling batch inference
        results = results[0]
        print("The results received in format output", results)
        for output in self._outputs:
            if isinstance(output, dict):
                if output["name"] in results.keys():
                    output_dict["name"] = output["name"]
                    output_dict["shape"] = [1] #static shape should be replaced with result shape
                    output_dict["datatype"] = "FP32" #Static types should be replaced with types based on result
                    output_dict["data"] = [results[output["name"]]]
                    outputs_list.append(output_dict)
                else :
                    print(f"The request key {output['name']} is not present in the prediction")
        
        
        response = {}
        response["outputs"] = outputs_list
        print("The Response of KFServing", response)
        return [response]