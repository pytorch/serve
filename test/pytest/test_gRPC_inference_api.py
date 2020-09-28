from ast import literal_eval
import inference_pb2
import json
import management_pb2
import os
import test_gRPC_utils
import test_utils


inference_data_json = "/../postman/inference_data.json"


def setup_module(module):
    test_utils.torchserve_cleanup()
    test_utils.start_torchserve()


def teardown_module(module):
    test_utils.torchserve_cleanup()


def __get_change(current, previous):
    if current == previous:
        return 0
    try:
        return (abs(current - previous) / previous) * 100.0
    except ZeroDivisionError:
        return float('inf')


def __infer(stub, model_name, model_input):
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
        managment_stub = test_gRPC_utils.get_management_stub()
        response = managment_stub.RegisterModel(management_pb2.RegisterModelRequest(
            url=item['url'],
            initial_workers=item['worker'],
            synchronous=bool(item['synchronous']),
            model_name=item['model_name']
        ))

        print(response.msg)

        model_input = os.path.dirname(__file__) + "/../" + item['file']
        prediction = __infer(test_gRPC_utils.get_inference_stub(), item['model_name'], model_input)

        print("Prediction is : ", str(prediction))

        if 'expected' in item:
            try:
                prediction = literal_eval(prediction)
            except SyntaxError:
                pass

            if isinstance(prediction, list) and 'tolerance' in item:
                assert len(prediction) == len(item['expected'])
                for i in range(len(prediction)):
                    assert __get_change(prediction[i], item['expected'][i]) < item['tolerance']
            elif isinstance(prediction, dict) and 'tolerance' in item:
                assert len(prediction) == len(item['expected'])
                for key in prediction:
                    assert __get_change(prediction[key], item['expected'][key]) < item['tolerance']
            else:
                assert str(prediction) == str(item['expected'])

        response = managment_stub.UnregisterModel(management_pb2.UnregisterModelRequest(
            model_name=item['model_name'],
        ))

        print(response.msg)
