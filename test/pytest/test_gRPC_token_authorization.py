import json
import os

import inference_pb2
import management_pb2
import pytest
import test_gRPC_utils
import test_utils

CURR_DIR = os.path.dirname(os.path.abspath(__file__))


def setup_module(module):
    test_utils.torchserve_cleanup()
    test_utils.start_torchserve(disable_token=False, gen_mar=False)


def teardown_module(module):
    test_utils.torchserve_cleanup()


def register(stub, model_name, metadata):
    marfile = f"https://torchserve.s3.amazonaws.com/mar_files/{model_name}.mar"
    params = {
        "url": marfile,
        "initial_workers": 1,
        "synchronous": True,
        "model_name": model_name,
    }
    stub.RegisterModel(
        management_pb2.RegisterModelRequest(**params), metadata=metadata, timeout=120
    )


def unregister(stub, model_name, metadata):
    params = {
        "model_name": model_name,
    }
    stub.UnregisterModel(
        management_pb2.UnregisterModelRequest(**params), metadata=metadata, timeout=60
    )


def infer(stub, model_name, model_input, metadata):
    with open(model_input, "rb") as f:
        data = f.read()

    input_data = {"data": data}
    params = {"model_name": model_name, "input": input_data}
    response = stub.Predictions(
        inference_pb2.PredictionsRequest(**params), metadata=metadata, timeout=60
    )

    return response.prediction.decode("utf-8")


def read_key_file(type):
    json_file_path = os.path.join(CURR_DIR, "key_file.json")
    with open(json_file_path) as json_file:
        json_data = json.load(json_file)

    options = {
        "management": json_data.get("management", {}).get("key", "NOT_PRESENT"),
        "inference": json_data.get("inference", {}).get("key", "NOT_PRESENT"),
        "token": json_data.get("API", {}).get("key", "NOT_PRESENT"),
    }
    key = options.get(type, "Invalid data type")
    return key


def test_grpc_api_with_token_auth():
    management_stub = test_gRPC_utils.get_management_stub()
    inference_stub = test_gRPC_utils.get_inference_stub()
    management_key = read_key_file("management")
    inference_key = read_key_file("inference")

    # register model with incorrect authorization token
    metadata = (("protocol", "gRPC"), ("authorization", f"Bearer incorrect-token"))
    with pytest.raises(Exception, match=r".*Token Authorization failed.*"):
        register(management_stub, "densenet161", metadata)

    # register model with correct authorization token
    metadata = (("protocol", "gRPC"), ("authorization", f"Bearer {management_key}"))
    register(management_stub, "densenet161", metadata)

    # make inference request with incorrect auth token
    metadata = (("protocol", "gRPC"), ("authorization", f"Bearer incorrect-token"))
    with pytest.raises(Exception, match=r".*Token Authorization failed.*"):
        infer(
            inference_stub,
            "densenet161",
            os.path.join(test_utils.REPO_ROOT, "examples/image_classifier/kitten.jpg"),
            metadata,
        )

    # make inference request with correct auth token
    metadata = (("protocol", "gRPC"), ("authorization", f"Bearer {inference_key}"))
    infer(
        inference_stub,
        "densenet161",
        os.path.join(test_utils.REPO_ROOT, "examples/image_classifier/kitten.jpg"),
        metadata,
    )

    # unregister model with incorrect authorization token
    metadata = (("protocol", "gRPC"), ("authorization", f"Bearer incorrect-token"))
    with pytest.raises(Exception, match=r".*Token Authorization failed.*"):
        unregister(management_stub, "densenet161", metadata)

    # unregister model with correct authorization token
    metadata = (("protocol", "gRPC"), ("authorization", f"Bearer {management_key}"))
    unregister(management_stub, "densenet161", metadata)
