import shutil
import sys
import threading
import time
from pathlib import Path

import pytest
import requests
import test_utils
from model_archiver.model_archiver_config import ModelArchiverConfig

CURR_FILE_PATH = Path(__file__).parent
STATEFUL_PATH = CURR_FILE_PATH.parents[1] / "examples" / "stateful"
STATEFUL_SEQUENCE_CONTINUOUS_PATH = (
    CURR_FILE_PATH.parents[1] / "examples" / "stateful" / "sequence_continuous_batching"
)
CONFIG_PROPERTIES_PATH = CURR_FILE_PATH.parents[1] / "test" / "config_ts.properties"

YAML_CONFIG = f"""
# TorchServe frontend parameters
minWorkers: 2
maxWorkers: 2
batchSize: 1
maxNumSequence: 2
sequenceMaxIdleMSec: 5000
maxSequenceJobQueueSize: 10
sequenceBatching: true
continuousBatching: true

handler:
  cache:
    capacity: 4
"""

JSON_INPUT = {
    "input": 3,
}


@pytest.fixture
def add_paths():
    sys.path.append(STATEFUL_SEQUENCE_CONTINUOUS_PATH.as_posix())
    yield
    sys.path.pop()


@pytest.fixture(scope="module")
def model_name():
    yield "stateful"


@pytest.fixture(scope="module")
def work_dir(tmp_path_factory, model_name):
    return tmp_path_factory.mktemp(model_name)


@pytest.fixture(scope="module", name="mar_file_path")
def create_mar_file(work_dir, model_archiver, model_name, request):
    mar_file_path = Path(work_dir).joinpath(model_name)

    model_config_yaml = Path(work_dir) / "model-config.yaml"
    model_config_yaml.write_text(YAML_CONFIG)

    config = ModelArchiverConfig(
        model_name=model_name,
        version="1.0",
        handler=(STATEFUL_SEQUENCE_CONTINUOUS_PATH / "stateful_handler.py").as_posix(),
        serialized_file=(STATEFUL_PATH / "model_cnn.pt").as_posix(),
        model_file=(STATEFUL_PATH / "model.py").as_posix(),
        export_path=work_dir,
        requirements_file=(STATEFUL_PATH / "requirements.txt").as_posix(),
        runtime="python",
        force=False,
        config_file=model_config_yaml.as_posix(),
        archive_format="no-archive",
    )

    model_archiver.generate_model_archive(config)

    assert mar_file_path.exists()

    yield mar_file_path.as_posix()

    # Clean up files
    shutil.rmtree(mar_file_path)


def test_infer_stateful(mar_file_path, model_store):
    """
    Register the model in torchserve
    """

    file_name = Path(mar_file_path).name

    model_name = Path(file_name).stem

    shutil.copytree(mar_file_path, Path(model_store) / model_name)

    params = (
        ("model_name", model_name),
        ("url", Path(model_store) / model_name),
        ("initial_workers", "2"),
        ("synchronous", "true"),
    )

    test_utils.start_torchserve(
        model_store=model_store, snapshot_file=CONFIG_PROPERTIES_PATH, gen_mar=False
    )

    try:
        test_utils.reg_resp = test_utils.register_model_with_params(params)

        t0 = threading.Thread(
            target=__infer_stateful,
            args=(
                model_name,
                "seq_0",
                "2 6 12 20 30",
            ),
        )
        t1 = threading.Thread(
            target=__infer_stateful,
            args=(
                model_name,
                "seq_1",
                "4 12 24 40 60",
            ),
        )

        t0.start()
        t1.start()

        t0.join()
        t1.join()
    finally:
        test_utils.unregister_model(model_name)

        # Clean up files
        shutil.rmtree(Path(model_store) / model_name)
        test_utils.stop_torchserve()


def test_infer_stateful_end(mar_file_path, model_store):
    """
    Register the model in torchserve
    """

    file_name = Path(mar_file_path).name

    model_name = Path(file_name).stem

    shutil.copytree(mar_file_path, Path(model_store) / model_name)

    params = (
        ("model_name", model_name),
        ("url", Path(model_store) / model_name),
        ("initial_workers", "2"),
        ("synchronous", "true"),
    )

    test_utils.start_torchserve(
        model_store=model_store, snapshot_file=CONFIG_PROPERTIES_PATH, gen_mar=False
    )

    try:
        test_utils.reg_resp = test_utils.register_model_with_params(params)

        t0 = threading.Thread(
            target=__infer_stateful_end,
            args=(
                model_name,
                "seq_0",
                "2 6 12 20 20",
            ),
        )
        t1 = threading.Thread(
            target=__infer_stateful,
            args=(
                model_name,
                "seq_1",
                "4 12 24 40 60",
            ),
        )

        t0.start()
        t1.start()

        t0.join()
        t1.join()
    finally:
        test_utils.unregister_model(model_name)

        # Clean up files
        shutil.rmtree(Path(model_store) / model_name)
        test_utils.stop_torchserve()


def test_infer_stateful_cancel(mar_file_path, model_store):
    """
    Register the model in torchserve
    """

    file_name = Path(mar_file_path).name

    model_name = Path(file_name).stem

    shutil.copytree(mar_file_path, Path(model_store) / model_name)

    params = (
        ("model_name", model_name),
        ("url", Path(model_store) / model_name),
        ("initial_workers", "2"),
        ("synchronous", "true"),
    )

    test_utils.start_torchserve(
        model_store=model_store, snapshot_file=CONFIG_PROPERTIES_PATH, gen_mar=False
    )

    try:
        test_utils.reg_resp = test_utils.register_model_with_params(params)
        with requests.post(
            url=f"http://localhost:8080/predictions/{model_name}",
            data=str(2).encode(),
        ) as response:
            s_id = response.headers.get("ts_request_sequence_id")
            headers = {
                "ts_request_sequence_id": s_id,
            }

        t0 = threading.Thread(
            target=__infer_stateful_cancel,
            args=(
                model_name,
                False,
                headers,
                "5",
            ),
        )
        t1 = threading.Thread(
            target=__infer_stateful_cancel,
            args=(
                model_name,
                True,
                headers,
                "-1",
            ),
        )

        t0.start()
        t1.start()

        t0.join()
        t1.join()
    finally:
        test_utils.unregister_model(model_name)

        # Clean up files
        shutil.rmtree(Path(model_store) / model_name)
        test_utils.stop_torchserve()


def __infer_stateful(model_name, sequence_id, expected):
    start = True
    prediction = []
    for idx in range(5):
        if sequence_id == "seq_0":
            idx = 2 * (idx + 1)
        elif sequence_id == "seq_1":
            idx = 4 * (idx + 1)
        if start is True:
            with requests.post(
                url=f"http://localhost:8080/predictions/{model_name}",
                data=str(idx).encode(),
            ) as response:
                s_id = response.headers.get("ts_request_sequence_id")
                if sequence_id == "seq_0":
                    headers_seq_0 = {
                        "ts_request_sequence_id": s_id,
                    }
                elif sequence_id == "seq_1":
                    headers_seq_1 = {
                        "ts_request_sequence_id": s_id,
                    }
                start = False
                prediction.append(response.text)
        else:
            with requests.post(
                url=f"http://localhost:8080/predictions/{model_name}",
                headers=headers_seq_0 if sequence_id == "seq_0" else headers_seq_1,
                data=str(idx).encode(),
            ) as response:
                prediction.append(response.text)

    print(f"infer_stateful prediction={str(' '.join(prediction))}")
    assert str(" ".join(prediction)) == expected


def __infer_stateful_end(model_name, sequence_id, expected):
    prediction = []
    start = True
    end = False
    for idx in range(5):
        if idx == 4:
            end = True
        if sequence_id == "seq_0":
            idx = 2 * (idx + 1)
        elif sequence_id == "seq_1":
            idx = 4 * (idx + 1)
        if end is True:
            idx = 0

        if start is True:
            with requests.post(
                url=f"http://localhost:8080/predictions/{model_name}",
                data=str(idx).encode(),
            ) as response:
                s_id = response.headers.get("ts_request_sequence_id")
                if sequence_id == "seq_0":
                    headers_seq_0 = {
                        "ts_request_sequence_id": s_id,
                    }
                elif sequence_id == "seq_1":
                    headers_seq_1 = {
                        "ts_request_sequence_id": s_id,
                    }
                start = False
                prediction.append(response.text)
        else:
            with requests.post(
                url=f"http://localhost:8080/predictions/{model_name}",
                headers=headers_seq_0 if sequence_id == "seq_0" else headers_seq_1,
                data=str(idx).encode(),
            ) as response:
                prediction.append(response.text)

    print(f"infer_stateful_end prediction={str(' '.join(prediction))}")
    assert str(" ".join(prediction)) == expected


def __infer_stateful_cancel(model_name, is_cancel, headers, expected):
    prediction = []
    if is_cancel:
        time.sleep(1)
        with requests.post(
            url=f"http://localhost:8080/predictions/{model_name}",
            headers=headers,
            data=str(-1).encode(),
        ) as response:
            prediction.append(response.text)
            print(f"infer_stateful_cancel prediction={str(' '.join(prediction))}")
            assert str(" ".join(prediction)) == expected
    else:
        with requests.post(
            url=f"http://localhost:8080/predictions/{model_name}",
            headers=headers,
            json=JSON_INPUT,
            stream=True,
        ) as response:
            assert response.headers["Transfer-Encoding"] == "chunked"
            for chunk in response.iter_content(chunk_size=None):
                if chunk:
                    prediction += [chunk.decode("utf-8")]

            print(f"infer_stateful_cancel prediction={str(' '.join(prediction))}")
            assert prediction[0] == expected
            assert len(prediction) < 11
