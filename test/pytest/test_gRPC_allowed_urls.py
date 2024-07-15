import os

import management_pb2
import pytest
import test_gRPC_utils
import test_utils

CONFIG_FILE = test_utils.ROOT_DIR + "/config.properties"


def setup_module(module):
    test_utils.torchserve_cleanup()
    with open(CONFIG_FILE, "w") as f:
        f.write(
            "allowed_urls=https://torchserve.s3.amazonaws.com/mar_files/densenet161.mar"
        )
    test_utils.start_torchserve(snapshot_file=CONFIG_FILE, gen_mar=False)


def teardown_module(module):
    test_utils.torchserve_cleanup()
    if os.path.exists(CONFIG_FILE):
        os.remove(CONFIG_FILE)


def register(stub, model_url, model_name, metadata):
    params = {
        "url": model_url,
        "initial_workers": 1,
        "synchronous": True,
        "model_name": model_name,
    }
    stub.RegisterModel(
        management_pb2.RegisterModelRequest(**params), metadata=metadata, timeout=120
    )


def test_gRPC_allowed_urls():
    management_stub = test_gRPC_utils.get_management_stub()

    # register model with permitted url
    metadata = (("protocol", "gRPC"),)
    register(
        management_stub,
        "https://torchserve.s3.amazonaws.com/mar_files/densenet161.mar",
        "densenet161",
        metadata,
    )

    # register model with unpermitted url
    with pytest.raises(
        Exception, match=r".*Given URL.*does not match any allowed URL.*"
    ):
        register(
            management_stub,
            "https://torchserve.s3.amazonaws.com/mar_files/resnet-18.mar",
            "resnet-18",
            metadata,
        )


def test_gRPC_allowed_urls_relative_path():
    management_stub = test_gRPC_utils.get_management_stub()

    # register model with relative path in model url
    metadata = (("protocol", "gRPC"),)
    with pytest.raises(Exception, match=r".*Relative path is not allowed in url.*"):
        register(
            management_stub,
            "https://torchserve.s3.amazonaws.com/mar_files/../mar_files/densenet161.mar",
            "densenet161-relative-path",
            metadata,
        )
