import shutil
from argparse import Namespace
from pathlib import Path

import pytest
import requests
import test_utils
from torchvision.models.resnet import ResNet18_Weights

from ts.torch_handler.unit_tests.test_utils.model_dir import download_model

CURR_FILE_PATH = Path(__file__).parent
REPO_ROOT_DIR = CURR_FILE_PATH.parent.parent

EXAMPLE_ROOT_DIR = REPO_ROOT_DIR.joinpath("examples", "microbatching")


@pytest.fixture(scope="module")
def model_name():
    yield "image_classifier"


@pytest.fixture(scope="module")
def work_dir(tmp_path_factory, model_name):
    return tmp_path_factory.mktemp(model_name)


@pytest.fixture(scope="module")
def serialized_file(work_dir):
    model_url = ResNet18_Weights.DEFAULT.url

    download_model(model_url, work_dir)

    yield Path(work_dir) / "model.pt"


@pytest.fixture(scope="module", name="mar_file_path")
def create_mar_file(
    work_dir, session_mocker, serialized_file, model_archiver, model_name
):
    mar_file_path = Path(work_dir).joinpath(model_name + ".mar")

    args = Namespace(
        model_name=model_name,
        version="1.0",
        serialized_file=str(serialized_file),
        model_file=REPO_ROOT_DIR.joinpath(
            "examples", "image_classifier", "resnet_18", "model.py"
        ).as_posix(),
        handler=REPO_ROOT_DIR.joinpath(
            "ts", "torch_handler", "composable_handler.py"
        ).as_posix(),
        extra_files="",
        export_path=work_dir,
        requirements_file=None,
        runtime="python",
        force=False,
        archive_format="default",
    )

    mock = session_mocker.MagicMock()
    mock.parse_args = session_mocker.MagicMock(return_value=args)
    session_mocker.patch(
        "archiver.ArgParser.export_model_args_parser", return_value=mock
    )

    # Using ZIP_STORED instead of ZIP_DEFLATED reduces test runtime from 54 secs to 10 secs
    from zipfile import ZIP_STORED, ZipFile

    session_mocker.patch(
        "model_archiver.model_packaging_utils.zipfile.ZipFile",
        lambda x, y, _: ZipFile(x, y, ZIP_STORED),
    )

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

    test_utils.reg_resp = test_utils.register_model(model_name, file_name)

    yield model_name

    test_utils.unregister_model(model_name)


def test_inference(model_name):
    """
    Full circle test with torchserve
    """

    response = requests.post(url=f"http://localhost:8080/predictions/{model_name}")

    assert response.status_code == 200
