import json
import os
import threading
from ast import literal_eval

import inference_pb2
import management_pb2
import test_gRPC_utils
import test_utils

inference_data_json = "../postman/inference_data.json"
inference_stream_data_json = "../postman/inference_stream_data.json"
inference_stream2_data_json = "../postman/inference_stream2_data.json"
config_file = test_utils.ROOT_DIR + "/config.properties"


def setup_module(module):
    test_utils.torchserve_cleanup()
    with open(config_file, "w") as f:
        f.write("install_py_dep_per_model=true")
    test_utils.start_torchserve(snapshot_file=config_file)


def teardown_module(module):
    test_utils.torchserve_cleanup()
    if os.path.exists(config_file):
        os.remove(config_file)


def __get_change(current, previous):
    if current == previous:
        return 0
    try:
        return (abs(current - previous) / previous) * 100.0
    except ZeroDivisionError:
        return float("inf")


def __infer(stub, model_name, model_input):
    with open(model_input, "rb") as f:
        data = f.read()

    input_data = {"data": data}
    response = stub.Predictions(
        inference_pb2.PredictionsRequest(model_name=model_name, input=input_data)
    )

    prediction = response.prediction.decode("utf-8")

    return prediction


def test_inference_apis():
    with open(os.path.join(os.path.dirname(__file__), inference_data_json), "rb") as f:
        test_data = json.loads(f.read())

    for item in test_data:
        # TODO: enable after correctly handling parameter name and header dtype in cpp backend
        if "skip_grpc_inference_api" in item and item["skip_grpc_inference_api"]:
            print(f"Skipping grpc inference api test for {item['url']}")
            continue

        if item["url"].startswith("{{mar_path_"):
            path = test_utils.mar_file_table[item["url"][2:-2]]
        else:
            path = item["url"]

        managment_stub = test_gRPC_utils.get_management_stub()
        response = managment_stub.RegisterModel(
            management_pb2.RegisterModelRequest(
                url=path,
                initial_workers=item["worker"],
                synchronous=bool(item["synchronous"]),
                model_name=item["model_name"],
            )
        )

        print(response.msg)

        model_input = os.path.join(os.path.dirname(__file__), "..", item["file"])
        prediction = __infer(
            test_gRPC_utils.get_inference_stub(), item["model_name"], model_input
        )

        print("Prediction is : ", str(prediction))

        if "expected" in item:
            try:
                prediction = literal_eval(prediction)
            except SyntaxError:
                pass

            if isinstance(prediction, list) and "tolerance" in item:
                assert len(prediction) == len(item["expected"])
                for i in range(len(prediction)):
                    assert (
                        __get_change(prediction[i], item["expected"][i])
                        < item["tolerance"]
                    )
            elif isinstance(prediction, dict) and "tolerance" in item:
                assert len(prediction) == len(item["expected"])
                for key in prediction:
                    assert (
                        __get_change(prediction[key], item["expected"][key])
                        < item["tolerance"]
                    )
            else:
                assert str(prediction) == str(item["expected"])

        response = managment_stub.UnregisterModel(
            management_pb2.UnregisterModelRequest(
                model_name=item["model_name"],
            )
        )

        print(response.msg)


def __infer_stream(stub, model_name, model_input):
    with open(model_input, "rb") as f:
        data = f.read()

    input_data = {"data": data}
    responses = stub.StreamPredictions(
        inference_pb2.PredictionsRequest(model_name=model_name, input=input_data)
    )

    prediction = []
    for resp in responses:
        prediction.append(resp.prediction.decode("utf-8"))

    return " ".join(prediction)


def test_inference_stream_apis():
    with open(
        os.path.join(os.path.dirname(__file__), inference_stream_data_json), "rb"
    ) as f:
        test_data = json.loads(f.read())

    for item in test_data:
        if item["url"].startswith("{{mar_path_"):
            path = test_utils.mar_file_table[item["url"][2:-2]]
        else:
            path = item["url"]

        managment_stub = test_gRPC_utils.get_management_stub()
        response = managment_stub.RegisterModel(
            management_pb2.RegisterModelRequest(
                url=path,
                initial_workers=item["worker"],
                synchronous=bool(item["synchronous"]),
                model_name=item["model_name"],
            )
        )

        print(response.msg)

        model_input = os.path.join(os.path.dirname(__file__), "..", item["file"])
        prediction = __infer_stream(
            test_gRPC_utils.get_inference_stub(), item["model_name"], model_input
        )

        print("Stream prediction is : ", str(prediction))

        if "expected" in item:
            assert str(prediction) == str(item["expected"])

        response = managment_stub.UnregisterModel(
            management_pb2.UnregisterModelRequest(
                model_name=item["model_name"],
            )
        )

        print(response.msg)


def __infer_stream2(stub, model_name, sequence_id, expected):
    request_iterator = iter(
        [
            inference_pb2.PredictionsRequest(
                model_name=model_name, input={"data": 1}, sequence_id=sequence_id
            ),
            inference_pb2.PredictionsRequest(
                model_name=model_name, input={"data": 2}, sequence_id=sequence_id
            ),
            inference_pb2.PredictionsRequest(
                model_name=model_name, input={"data": 3}, sequence_id=sequence_id
            ),
        ]
    )
    responses_iterator = stub.StreamPredictions2(request_iterator)

    prediction = []
    for resp in responses_iterator:
        prediction.append(resp.prediction.decode("utf-8"))

    assert str(" ".join(prediction)) == expected


def test_inference_stream2_apis():
    with open(
        os.path.join(os.path.dirname(__file__), inference_stream2_data_json), "rb"
    ) as f:
        test_data = json.loads(f.read())

    for item in test_data:
        model_artifacts = test_utils.create_model_artifacts(item, force=True)
        managment_stub = test_gRPC_utils.get_management_stub()
        response = managment_stub.RegisterModel(
            management_pb2.RegisterModelRequest(
                url=model_artifacts,
                initial_workers=item["worker"],
                synchronous=bool(item["synchronous"]),
                model_name=item["model_name"],
            )
        )

        print(response.msg)

        t0 = threading.Thread(
            target=__infer_stream2,
            args=(
                test_gRPC_utils.get_inference_stub(),
                item["model_name"],
                "seq_0",
                str(item["expected"]),
            ),
        )
        t1 = threading.Thread(
            target=__infer_stream2,
            args=(
                test_gRPC_utils.get_inference_stub(),
                item["model_name"],
                "seq_1",
                str(item["expected"]),
            ),
        )

        t0.start()
        t1.start()

        t0.join()
        t1.join()

        response = managment_stub.UnregisterModel(
            management_pb2.UnregisterModelRequest(
                model_name=item["model_name"],
            )
        )

        print(response.msg)
