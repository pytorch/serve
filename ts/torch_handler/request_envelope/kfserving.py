import json
from itertools import chain
from base64 import b64decode

from .base import BaseEnvelope

class KFservingEnvelope(BaseEnvelope):
    """
    Implementation. Captures batches in JSON format, returns
    also in JSON format.
    """
    def parse_input(self, data):
        print("Parsing input in KFServing.py")
        self._data_list = [row.get("data") or row.get("body") for row in data]
        #selecting the first input from the list torchserve creates
        print("Parse input data_list ",self._data_list)
        data = self._data_list[0]
        
        #IF the KF Transformer and Explainer sends in data as bytesarray 
        if isinstance(data, (bytes, bytearray)):
            
            data = data.decode()
            data = json.loads(data)
            print("Bytes array is ",data)
        # try:
        #     data = data.decode()
        #     print("Bytes array is ",data)
        # except (UnicodeDecodeError, AttributeError):
        #     pass
        
        self._inputs = data.get("instances") 
        #selecting the first input from the kfserving request instances list
        #self._inputs = self._inputs[0]
        print("KFServing parsed inputs", self._inputs)
        return self._inputs

    def format_output(self, outputs):
        output = outputs[0] #Removing the outer list added in base handler for consistency
        response = {}
        response["predictions"] = output["predictions"]
        if "explanations" in output:
            response["explanations"] =  output["explanations"]
  
        print("The Response of KFServing", response)
        return [response]