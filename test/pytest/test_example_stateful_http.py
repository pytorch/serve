import json
import os
import threading

import requests
import test_utils

inference_stream2_data_json = "../postman/inference_stream2_data.json"
config_file = test_utils.ROOT_DIR + "/config.properties"
with open(config_file, "w") as f:
    f.write("install_py_dep_per_model=true")


def setup_module(module):
    test_utils.torchserve_cleanup()
    test_utils.start_torchserve(snapshot_file=config_file)


def teardown_module(module):
    test_utils.torchserve_cleanup()


def __infer_stateful(model_name, sequence_id, expected):
    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "ts_request_sequence_id": sequence_id,
    }
    prediction = []
    for idx in range(3):
        response = requests.post(
            url=f"http://localhost:8080/predictions/{model_name}",
            headers=headers,
            json={"data": idx},
        )
        prediction.append(response.text)

    assert str(" ".join(prediction)) == expected


def test_example_stateful_http():
    with open(
        os.path.join(os.path.dirname(__file__), inference_stream2_data_json), "rb"
    ) as f:
        test_data = json.loads(f.read())

    for item in test_data:
        model_artifacts = test_utils.create_model_artifacts(item, force=True)
        params = (
            ("model_name", item["model_name"]),
            ("url", model_artifacts),
            ("initial_workers", item["worker"]),
            ("synchronous", "true"),
        )

        try:
            test_utils.reg_resp = test_utils.register_model_with_params(params)

            t0 = threading.Thread(
                target=__infer_stateful,
                args=(
                    item["model_name"],
                    "seq_0",
                    str(item["expected"]),
                ),
            )
            t1 = threading.Thread(
                target=__infer_stateful,
                args=(
                    item["model_name"],
                    "seq_1",
                    str(item["expected"]),
                ),
            )

            t0.start()
            t1.start()

            t0.join()
            t1.join()
        finally:
            test_utils.unregister_model(item["model_name"])
