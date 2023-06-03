import shutil
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch
from zipfile import ZIP_STORED, ZipFile

import pytest
import requests
import test_utils

CURR_FILE_PATH = Path(__file__).parent


@pytest.fixture(scope="module")
def model_name():
    yield "streaming_handler"


@pytest.fixture(scope="module")
def work_dir(tmp_path_factory, model_name):
    return tmp_path_factory.mktemp(model_name)


@pytest.fixture(scope="module", name="mar_file_path")
def create_mar_file(work_dir, model_archiver, model_name):
    mar_file_path = Path(work_dir).joinpath(model_name + ".mar")

    args = Namespace(
        model_name=model_name,
        version="1.0",
        model_file=CURR_FILE_PATH.joinpath(
            "test_data", "fake_streaming_model.py"
        ).as_posix(),
        handler=CURR_FILE_PATH.joinpath("test_data", "stream_handler.py").as_posix(),
        serialized_file=None,
        export_path=work_dir,
        requirements_file=None,
        runtime="python",
        force=False,
        archive_format="default",
        config_file=None,
        extra_files=None,
    )

    mock = MagicMock()
    mock.parse_args = MagicMock(return_value=args)
    with patch("archiver.ArgParser.export_model_args_parser", return_value=mock):
        # Using ZIP_STORED instead of ZIP_DEFLATED reduces test runtime from 54 secs to 10 secs
        with patch(
            "model_archiver.model_packaging_utils.zipfile.ZipFile",
            lambda x, y, _: ZipFile(x, y, ZIP_STORED),
        ):
            model_archiver.generate_model_archive()

            assert mar_file_path.exists()

            yield mar_file_path.as_posix()

    # Clean up files
    # mar_file_path.unlink(missing_ok=True)


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
        ("batch_size", "2"),
    )

    test_utils.reg_resp = test_utils.register_model_with_params(params)

    yield model_name

    test_utils.unregister_model(model_name)


def test_echo_stream_inference(model_name):
    responses = []

    for _ in range(2):
        res = requests.post(
            url=f"http://localhost:8080/predictions/{model_name}",
            data="foo",
            stream=True,
        )

        responses.append(res)

    assert all(r.headers["Transfer-Encoding"] == "chunked" for r in responses)

    for idx, _ in enumerate(responses):
        prediction = []
        for chunk in responses[idx].iter_content(chunk_size=None):
            if chunk:
                prediction.append(chunk.decode("utf-8"))

        assert (
            f"{idx}" + str("".join(prediction))
            == f"{idx}" + "hello hello hello hello world "
        )

    test_utils.unregister_model("echo_stream")
