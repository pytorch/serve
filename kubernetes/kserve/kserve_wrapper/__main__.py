""" KServe wrapper to handler inference in the kserve_predictor """
import json
import logging
import os

import kserve
from kserve.model_server import ModelServer
from TorchserveModel import PredictorProtocol, TorchserveModel
from TSModelRepository import TSModelRepository

logging.basicConfig(level=kserve.constants.KSERVE_LOGLEVEL)

DEFAULT_MODEL_NAME = "model"
DEFAULT_INFERENCE_ADDRESS = DEFAULT_MANAGEMENT_ADDRESS = "http://127.0.0.1:8085"
DEFAULT_GRPC_INFERENCE_PORT = "7070"

DEFAULT_MODEL_STORE = "/mnt/models/model-store"
DEFAULT_CONFIG_PATH = "/mnt/models/config/config.properties"


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
    config_path = os.environ.get("CONFIG_PATH", DEFAULT_CONFIG_PATH)

    logging.info(f"Wrapper: loading configuration from {config_path}")

    with open(config_path) as f:
        for line in f:
            if separator in line:
                # Find the name and value by splitting the string
                name, value = line.split(separator, 1)

                # Assign key value pair to dict
                # strip() removes white space from the ends of strings
                keys[name.strip()] = value.strip()

    keys["model_snapshot"] = json.loads(keys["model_snapshot"])
    inference_address, management_address, grpc_inference_port, model_store = (
        keys["inference_address"],
        keys["management_address"],
        keys["grpc_inference_port"],
        keys["model_store"],
    )

    models = keys["model_snapshot"]["models"]
    model_names = []

    # Get all the model_names
    for model, value in models.items():
        model_names.append(model)

    if not inference_address:
        inference_address = DEFAULT_INFERENCE_ADDRESS
    if not model_names:
        model_names = [DEFAULT_MODEL_NAME]
    if not inference_address:
        inference_address = DEFAULT_INFERENCE_ADDRESS
    if not management_address:
        management_address = DEFAULT_MANAGEMENT_ADDRESS
    inf_splits = inference_address.split(":")
    if not grpc_inference_port:
        grpc_inference_address = inf_splits[1] + ":" + DEFAULT_GRPC_INFERENCE_PORT
    else:
        grpc_inference_address = inf_splits[1] + ":" + grpc_inference_port
    grpc_inference_address = grpc_inference_address.replace("/", "")
    if not model_store:
        model_store = DEFAULT_MODEL_STORE

    logging.info(
        "Wrapper : Model names %s, inference address %s, management address %s, grpc_inference_address, %s, model store %s",
        model_names,
        inference_address,
        management_address,
        grpc_inference_address,
        model_store,
    )

    return (
        model_names,
        inference_address,
        management_address,
        grpc_inference_address,
        model_store,
    )


if __name__ == "__main__":
    (
        model_names,
        inference_address,
        management_address,
        grpc_inference_address,
        model_dir,
    ) = parse_config()

    protocol = os.environ.get("PROTOCOL_VERSION", PredictorProtocol.REST_V1.value)

    models = []

    for model_name in model_names:
        model = TorchserveModel(
            model_name,
            inference_address,
            management_address,
            grpc_inference_address,
            protocol,
            model_dir,
        )
        # By default model.load() is called on first request. Enabling load all
        # model in TS config.properties, all models are loaded at start and the
        # below method sets status to true for the models.
        # However, even if all preparations related to loading the model (e.g.,
        # download pretrained models using online storage) are not completed in
        # torchserve handler, if model.ready=true is set, there may be problems.
        # Therefore, the ready status is determined using the api provided by
        # torchserve.

        model.load()
        models.append(model)

    registeredModels = TSModelRepository(
        inference_address,
        management_address,
        model_dir,
    )
    ModelServer(
        registered_models=registeredModels,
        http_port=8080,
        grpc_port=8081,
    ).start(models)
