import kfserving
import argparse
import json
import logging
logging.basicConfig(level=kfserving.constants.KFSERVING_LOGLEVEL)
from TorchserveModel import TorchserveModel
from TSModelRepository import TSModelRepository

DEFAULT_MODEL_NAME = "model"
DEFAULT_INFERENCE_ADDRESS = "http://127.0.0.1:8085"
INFERENCE_PORT = "8085"
DEFAULT_MANAGEMENT_ADDRESS = "http://127.0.0.1:8081"

DEFAULT_MODEL_STORE = "/mnt/models/model-store"
CONFIG_PATH = "/mnt/models/config/config.properties"

def parse_config():
    separator = "="
    keys = {}

    with open(CONFIG_PATH) as f:

        for line in f:
            if separator in line:

                # Find the name and value by splitting the string
                name, value = line.split(separator, 1)

                # Assign key value pair to dict
                # strip() removes white space from the ends of strings
                keys[name.strip()] = value.strip()
              
    keys['model_snapshot'] = json.loads(keys['model_snapshot'])
    inference_address, management_address, model_store = keys['inference_address'], keys['management_address'], keys["model_store"]
    
    models = keys['model_snapshot']['models']
    model_names = []
    
    #constructs inf address at a port other than 8080 as kfserver runs at 8080
    if inference_address != None:
        inf_splits = inference_address.split(":")
        inference_address = inf_splits[0]+inf_splits[1]+ ":" + INFERENCE_PORT
    else :
        inference_address = DEFAULT_INFERENCE_ADDRESS 
    #Get all the model_names
    for model, value in models.items():
        model_names.append(model)
    if not model_names:
        model_names = [DEFAULT_MODEL_NAME]
    if management_address == None:
        management_address = DEFAULT_MANAGEMENT_ADDRESS
    if model_store == None:
        model_store = DEFAULT_MODEL_STORE
    print(f"Wrapper : Model names {model_names}, inference_address {inference_address}, management_address {management_address} , model_store {model_store} ")
    return model_names, inference_address, management_address, model_store 

if __name__ == "__main__":
    #model = PyTorchModel(args.model_name, args.model_class_name, args.model_dir)
    # model_names = 
    # predictor_host = 
    model_names, inference_address, management_address, model_dir = parse_config()

    models = []

    for model_name in model_names:

        model = TorchserveModel(model_name,inference_address, management_address, model_dir)
        models.append(model)
    kfserving.KFServer(registered_models=TSModelRepository(inference_address, management_address, model_dir), http_port = 8080).start(models)
