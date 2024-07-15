import management_pb2
import pytest
import test_gRPC_utils
import test_utils


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


def test_grpc_register_model_with_model_control():
    test_utils.torchserve_cleanup()
    test_utils.start_torchserve(enable_model_api=False, gen_mar=False)
    management_stub = test_gRPC_utils.get_management_stub()

    metadata = (("protocol", "gRPC"),)
    with pytest.raises(Exception, match=r".*Model API disabled.*"):
        register(management_stub, "densenet161", metadata)

    test_utils.torchserve_cleanup()


def test_grpc_unregister_model_with_model_control():
    test_utils.torchserve_cleanup()
    test_utils.start_torchserve(
        enable_model_api=False,
        gen_mar=False,
        models="densenet161=https://torchserve.s3.amazonaws.com/mar_files/densenet161.mar",
    )
    management_stub = test_gRPC_utils.get_management_stub()

    metadata = (("protocol", "gRPC"),)
    with pytest.raises(Exception, match=r".*Model API disabled.*"):
        unregister(management_stub, "densenet161", metadata)

    test_utils.torchserve_cleanup()
