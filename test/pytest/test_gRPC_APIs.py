from ast import literal_eval
import grpc
import inference_pb2
import inference_pb2_grpc
import json
import os
import test_utils


inference_data_json = "/../postman/inference_data.json"
torchserve_management_url = "http://localhost:8081/models"


def setup_module(module):
    test_utils.torchserve_cleanup()
    test_utils.start_torchserve()


def teardown_module(module):
    test_utils.torchserve_cleanup()


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

    return prediction


def test_inference_apis():
    with open(os.path.dirname(__file__) + inference_data_json, 'rb') as f:
        test_data = json.loads(f.read())

    for item in test_data:
        params = (
            ('model_name', item['model_name']),
            ('url', item['url']),
            ('initial_workers', item['worker']),
            ('synchronous', item['synchronous']),
        )

        test_utils.register_model(params)

        model_input = os.path.dirname(__file__) + "/../" + item['file']
        prediction = infer(get_inference_stub(), item['model_name'], model_input)

        print("Prediction is : ", str(prediction))

        if 'expected' in item:
            try:
                prediction = literal_eval(prediction)
            except SyntaxError:
                pass

            assert str(prediction) == str(item['expected'])

        test_utils.unregister_model(item['model_name'])
