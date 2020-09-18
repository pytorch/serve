import inference_pb2
import inference_pb2_grpc
import grpc
import sys

number_of_requests = 1000


def get_inference_stub():
    channel = grpc.insecure_channel('localhost:9090')
    stub = inference_pb2_grpc.InferenceAPIsServiceStub(channel)
    return stub


def infer(stub, model_name, model_input):
    with open(model_input, 'rb') as f:
        data = f.read()

    input_data = {'data': data}
    response = stub.Predictions(
            inference_pb2.PredictionsRequest(model_name=model_name, input=input_data))

    prediction = response.prediction.decode('utf-8')

    if not prediction:
        print(str(response.status_code))
        print(str(response.info))
    else:
        print(response.prediction.decode('utf-8'))


if __name__ == '__main__':
    args = sys.argv[1:]
    infer(get_inference_stub(), args[0], args[1])
