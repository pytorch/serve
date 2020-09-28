import grpc
import inference_pb2_grpc
import management_pb2_grpc


def get_inference_stub():
    channel = grpc.insecure_channel('localhost:9090')
    stub = inference_pb2_grpc.InferenceAPIsServiceStub(channel)
    return stub


def get_management_stub():
    channel = grpc.insecure_channel('localhost:9091')
    stub = management_pb2_grpc.ManagementAPIsServiceStub(channel)
    return stub


def register_model():
    pass


def unregister_model():
    pass


def scale_model():
    pass


def set_default_model():
    pass


def list_model():
    pass


def describe_model():
    pass


