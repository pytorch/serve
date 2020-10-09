import grpc
import inference_pb2_grpc
import management_pb2
import management_pb2_grpc


def get_inference_stub():
    channel = grpc.insecure_channel('localhost:9090')
    stub = inference_pb2_grpc.InferenceAPIsServiceStub(channel)
    return stub


def get_management_stub():
    channel = grpc.insecure_channel('localhost:9091')
    stub = management_pb2_grpc.ManagementAPIsServiceStub(channel)
    return stub


def register_model(**kwargs):
    management_stub = get_management_stub()
    return management_stub.RegisterModel(management_pb2.RegisterModelRequest(**kwargs))


def unregister_model(**kwargs):
    management_stub = get_management_stub()
    return management_stub.UnregisterModel(management_pb2.UnregisterModelRequest(**kwargs))


def scale_model(**kwargs):
    management_stub = get_management_stub()
    return management_stub.ScaleWorker(management_pb2.ScaleWorkerRequest(**kwargs))


def set_default_model(**kwargs):
    management_stub = get_management_stub()
    return management_stub.SetDefault(management_pb2.SetDefaultRequest(**kwargs))


def list_model(**kwargs):
    management_stub = get_management_stub()
    return management_stub.ListModels(management_pb2.ListModelsRequest(**kwargs))


def describe_model(**kwargs):
    management_stub = get_management_stub()
    return management_stub.DescribeModel(management_pb2.DescribeModelRequest(**kwargs))
