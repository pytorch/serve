import shutil
import sys
import threading
from pathlib import Path

import pytest
import requests
import test_utils
from model_archiver.model_archiver_config import ModelArchiverConfig

CURR_FILE_PATH = Path(__file__).parent
STATEFUL_PATH = CURR_FILE_PATH.parents[1] / "examples" / "stateful"
CONFIG_PROPERTIES_PATH = CURR_FILE_PATH.parents[1] / "test" / "config_ts.properties"

YAML_CONFIG = f"""
# TorchServe frontend parameters
minWorkers: 2
maxWorkers: 2
batchSize: 4
maxNumSequence: 4
sequenceMaxIdleMSec: 10
maxSequenceJobQueueSize: 10
sequenceBatch: true

handler:
  cache:
    capacity: 4
"""

PROMPTS = [
    {
        "prompt": "A robot may not injure a human being",
        "max_new_tokens": 50,
        "temperature": 0.8,
        "logprobs": 1,
        "prompt_logprobs": 1,
        "max_tokens": 128,
        "adapter": "adapter_1",
    },
]


@pytest.fixture
def add_paths():
    sys.path.append(STATEFUL_PATH.as_posix())
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
        handler=(STATEFUL_PATH / "stateful_handler.py").as_posix(),
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


def test_stateful_mar(mar_file_path, model_store):
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
                "1 3 6",
            ),
        )
        t1 = threading.Thread(
            target=__infer_stateful,
            args=(
                model_name,
                "seq_1",
                "1 3 6",
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


def __infer_stateful(model_name, sequence_id, expected):
    headers = {
        "Content-Type": "text/plain",
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
