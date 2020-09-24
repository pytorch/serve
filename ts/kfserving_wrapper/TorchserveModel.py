import kfserving
import logging
logging.basicConfig(level=kfserving.constants.KFSERVING_LOGLEVEL)
import tornado.web
import os
import json
from typing import Dict
import requests
REGISTER_URL_FORMAT = "{0}/models?initial_workers=1&url={1}"
UNREGISTER_URL_FORMAT = "{0}/models/{1}"

PREDICTOR_URL_FORMAT = "http://{0}/v1/models/{1}:predict"
EXPLAINER_URL_FORMAT = "http://{0}/v1/models/{1}:explain"

class TorchserveModel(kfserving.KFModel):
    def __init__(self, name,inference_address, management_address, model_dir ):
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
        if not self.predictor_host:
            raise NotImplementedError
        print("kfmodel predict request is ",json.dumps(request))
        print("PREDICTOR_HOST :", self.predictor_host) 
        headers = {'Content-Type': 'application/json; charset=UTF-8'}
        response = await self._http_client.fetch(
            PREDICTOR_URL_FORMAT.format(self.predictor_host, self.name),
            method='POST',
            request_timeout=self.timeout,
            headers = headers,
            body= json.dumps(request)
        )
        
        if response.code != 200:
            raise tornado.web.HTTPError(
                status_code=response.code,
                reason=response.body)
        return json.loads(response.body)
    
    async def explain(self, request: Dict) -> Dict:
        if self.explainer_host is None:
            raise NotImplementedError
        print("kfmodel explain request is ",json.dumps(request)) 
        print("EXPLAINER_HOST :", self.explainer_host) 
        headers = {'Content-Type': 'application/json; charset=UTF-8'}  
        response = await self._http_client.fetch(
            EXPLAINER_URL_FORMAT.format(self.explainer_host, self.name),
            method='POST',
            request_timeout=self.timeout,
            headers = headers,
            body=json.dumps(request)
        )
        if response.code != 200:
            raise tornado.web.HTTPError(
                status_code=response.code,
                reason=response.body)
        return json.loads(response.body)

    def load_model(self) -> bool:
        if not self.management_address:
            raise NotImplementedError
        model_path = os.path.join(self.model_dir, self.name)
        print("kfmodel loading model ",self.name)
        # response = self._http_client.fetch(
        #     REGISTER_URL_FORMAT.format(self.management_address, model_path),
        #     method='POST',
        #     request_timeout=self.timeout,
        #     headers = headers,
        #     body=json.dumps({"None":"None"}))

        # if response != 200:
        #     raise tornado.web.HTTPError(
        #         status_code=response,
        #         reason=response.body)
       
        # else :
        #     self.ready = True
        # return self.ready
        model_name = f"{self.name}.mar"
        response = requests.post(	        
            REGISTER_URL_FORMAT.format(self.management_address, model_name))	  
        if response.status_code != 200:	        
            raise tornado.web.HTTPError(status_code=response.status_code,reason=response.content)	 
        else :
            self.ready = True
        return self.ready      

    def unload_model(self) -> bool:
        if not self.management_address:
            raise NotImplementedError
        #model_path = os.path.join(self.model_dir, self.name)
        print("kfmodel unloading model ",self.name)
        # headers = {'Content-Type': 'application/json; charset=UTF-8'}
        # response = self._http_client.fetch(
        #     UNREGISTER_URL_FORMAT.format(self.management_address, self.name),
        #     method='DELETE',
        #     request_timeout=self.timeout,
        #     headers = headers,
        #     body=None)
        # print("This is the response after unload", response)
        # if response != 200:
        #     raise tornado.web.HTTPError(response)
        # else :
        #     self.ready = False
        # return self.ready

        response = requests.delete(	        
            UNREGISTER_URL_FORMAT.format(self.management_address, self.name))	  
        if response.status_code != 200:	        
            raise tornado.web.HTTPError(status_code=response.status_code,reason=response.content)	 
        else :
            self.ready = False
        return self.ready    
                  
        