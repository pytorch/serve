from ast import literal_eval
import grpc
import inference_pb2
import inference_pb2_grpc
import json
import management_pb2
import management_pb2_grpc
import os
import test_utils


inference_data_json = "/../postman/inference_data.json"

def setup_module(module):
    test_utils.torchserve_cleanup()
    test_utils.start_torchserve()


def teardown_module(module):
    test_utils.torchserve_cleanup()


def get_inference_stub():
    channel = grpc.insecure_channel('localhost:9090')
    stub = inference_pb2_grpc.InferenceAPIsServiceStub(channel)
    return stub


def get_managment_stub():
    channel = grpc.insecure_channel('localhost:9091')
    stub = management_pb2_grpc.ManagementAPIsServiceStub(channel)
    return stub


def get_change(current, previous):
    if current == previous:
        return 0
    try:
        return (abs(current - previous) / previous) * 100.0
    except ZeroDivisionError:
        return float('inf')


def infer(stub, model_name, model_input):
    with open(model_input, 'rb') as f:
        data = f.read()

    input_data = {'data': data}
    response = stub.Predictions(
            inference_pb2.PredictionsRequest(model_name=model_name, input=input_data))

    prediction = response.prediction.decode('utf-8')

    return prediction


def test_inference_apis():
    with open(os.path.dirname(__file__) + inference_data_json, 'rb') as f:
        test_data = json.loads(f.read())

    for item in test_data:
        managment_stub = get_managment_stub()
        response = managment_stub.RegisterModel(management_pb2.RegisterModelRequest(
            url=item['url'],
            initial_workers=item['worker'],
            synchronous=bool(item['synchronous']),
            model_name=item['model_name']
        ))

        print(response.msg)

        model_input = os.path.dirname(__file__) + "/../" + item['file']
        prediction = infer(get_inference_stub(), item['model_name'], model_input)

        print("Prediction is : ", str(prediction))

        if 'expected' in item:
            try:
                prediction = literal_eval(prediction)
            except SyntaxError:
                pass

            if isinstance(prediction, list) and 'tolerance' in item:
                assert len(prediction) == len(item['expected'])
                for i in range(len(prediction)):
                    assert get_change(prediction[i], item['expected'][i]) < item['tolerance']
            elif isinstance(prediction, dict) and 'tolerance' in item:
                assert len(prediction) == len(item['expected'])
                for key in prediction:
                    assert get_change(prediction[key], item['expected'][key]) < item['tolerance']
            else:
                assert str(prediction) == str(item['expected'])

        response = managment_stub.UnregisterModel(management_pb2.UnregisterModelRequest(
            model_name=item['model_name'],
        ))

        print(response.msg)
