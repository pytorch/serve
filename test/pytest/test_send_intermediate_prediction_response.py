import json
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest
import requests
import test_utils
from model_archiver import ModelArchiverConfig

CURR_FILE_PATH = Path(__file__).parent
REPO_ROOT_DIR = CURR_FILE_PATH.parent.parent

HANDLER_PY = """
from ts.handler_utils.utils import send_intermediate_predict_response

def handle(data, context):
    if type(data) is list:
        for i in range (3):
            send_intermediate_predict_response(["hello"], context.request_ids, "Intermediate Prediction success", 200, context)
        return ["hello world "]

"""


@pytest.fixture(scope="module")
def model_name():
    yield "tp_model"


@pytest.fixture(scope="module")
def work_dir(tmp_path_factory, model_name):
    return Path(tmp_path_factory.mktemp(model_name))


@pytest.fixture(scope="module", name="mar_file_path")
def create_mar_file(work_dir, model_archiver, model_name):
    mar_file_path = work_dir.joinpath(model_name + ".mar")

    handler_py_file = work_dir / "handler.py"
    handler_py_file.write_text(HANDLER_PY)

    config = ModelArchiverConfig(
        model_name=model_name,
        version="1.0",
        serialized_file=None,
        model_file=None,
        handler=handler_py_file.as_posix(),
        extra_files=None,
        export_path=work_dir,
        requirements_file=None,
        runtime="python",
        force=False,
        archive_format="default",
        config_file=None,
    )

    with patch("archiver.ArgParser.export_model_args_parser", return_value=config):
        model_archiver.generate_model_archive()

        assert mar_file_path.exists()

        yield mar_file_path.as_posix()

    # Clean up files
    mar_file_path.unlink(missing_ok=True)


@pytest.fixture(scope="module", name="model_name")
def register_model(mar_file_path, model_store, torchserve):
    """
    Register the model in torchserve
    """
    shutil.copy(mar_file_path, model_store)

    file_name = Path(mar_file_path).name

    model_name = Path(file_name).stem

    params = (
        ("model_name", model_name),
        ("url", file_name),
        ("initial_workers", "1"),
        ("synchronous", "true"),
        ("batch_size", "1"),
    )

    test_utils.reg_resp = test_utils.register_model_with_params(params)

    yield model_name

    test_utils.unregister_model(model_name)


@pytest.mark.parametrize(("params"), ((True, 4), (False, 1)))
def test_echo_stream_inference(model_name, params):
    """
    Full circle test with torchserve
    """
    STREAM = params[0]
    EXPECTED_RESPONSES = params[1]

    response = requests.post(
        url=f"http://localhost:8080/predictions/{model_name}",
        data=json.dumps(42),
        stream=STREAM,
    )

    assert response.status_code == 200

    assert response.headers["Transfer-Encoding"] == "chunked"

    prediction = []
    for chunk in response.iter_content(chunk_size=None):
        if chunk:
            prediction += [chunk.decode("utf-8")]

    assert len(prediction) == EXPECTED_RESPONSES

    assert str("".join(prediction)) == "hellohellohellohello world "
