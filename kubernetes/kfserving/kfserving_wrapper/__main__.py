""" KServe wrapper to handler inference in the kserve_predictor """
import json
import logging
import kserve
from TorchserveModel import TorchserveModel
from TSModelRepository import TSModelRepository

logging.basicConfig(level=kserve.constants.KSERVE_LOGLEVEL)

DEFAULT_MODEL_NAME = "model"
DEFAULT_INFERENCE_ADDRESS = "http://127.0.0.1:8085"
INFERENCE_PORT = "8085"
DEFAULT_MANAGEMENT_ADDRESS = "http://127.0.0.1:8081"

DEFAULT_MODEL_STORE = "/mnt/models/model-store"
CONFIG_PATH = "/mnt/models/config/config.properties"


def parse_config():
    """This function parses the model snapshot from the config.properties file

    Returns:
        model_name: The name of the model specified in the config.properties
        inference_address: The inference address in which the inference endpoint is hit
        management_address: The management address in which the model gets registered
        model_store: the path in which the .mar file resides
    """
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

    keys["model_snapshot"] = json.loads(keys["model_snapshot"])
    inference_address, management_address, model_store = (
        keys["inference_address"],
        keys["management_address"],
        keys["model_store"],
    )

    models = keys["model_snapshot"]["models"]
    model_names = []

    # constructs inf address at a port other than 8080 as kfserver runs at 8080
    if inference_address:
        inf_splits = inference_address.split(":")
        inference_address = inf_splits[0] + inf_splits[1] + ":" + INFERENCE_PORT
    else:
        inference_address = DEFAULT_INFERENCE_ADDRESS
    # Get all the model_names
    for model, value in models.items():
        model_names.append(model)
    if not model_names:
        model_names = [DEFAULT_MODEL_NAME]
    if not management_address:
        management_address = DEFAULT_MANAGEMENT_ADDRESS
    if not model_store:
        model_store = DEFAULT_MODEL_STORE
    logging.info("Wrapper : Model names %s, inference address %s, management address %s, model store %s", model_names,
                 inference_address, management_address, model_store)

    return model_names, inference_address, management_address, model_store


if __name__ == "__main__":

    model_names, inference_address, management_address, model_dir = parse_config()

    models = []

    for model_name in model_names:

        model = TorchserveModel(model_name, inference_address, management_address, model_dir)
        models.append(model)
    kserve.KFServer(
        registered_models=TSModelRepository(inference_address, management_address, model_dir),
        http_port=8080,
        grpc_port=7070,
    ).start(models)
