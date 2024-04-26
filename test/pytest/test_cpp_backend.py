import io
import os
import shutil
from pathlib import Path
from unittest.mock import patch
from zipfile import ZIP_STORED, ZipFile

import pytest
import requests
import test_utils
import torch
from model_archiver import ModelArchiverConfig
from PIL import Image
from torchvision import transforms

import ts

CURR_FILE_PATH = Path(__file__).parent

TORCH_NCCL_PATH = (Path(torch.__file__).parent / "lib").as_posix()
TORCH_NCCL_PATH += (
    ":" + (Path(torch.__file__).parents[1] / "nvidia" / "nccl" / "lib").as_posix()
)
os.environ["LD_LIBRARY_PATH"] = (
    TORCH_NCCL_PATH + ":" + os.environ.get("LD_LIBRARY_PATH", "")
)


@pytest.fixture(scope="module")
def model_name():
    yield "mnist_handler"


@pytest.fixture(scope="module")
def work_dir(tmp_path_factory, model_name):
    return tmp_path_factory.mktemp(model_name)


@pytest.fixture(scope="module", name="mar_file_path")
def create_mar_file(work_dir, model_archiver, model_name):
    mar_file_path = Path(work_dir).joinpath(model_name + ".mar")

    mnist_scriptes_pt = (
        CURR_FILE_PATH.parents[1]
        / "cpp/build/test/resources/examples/mnist/mnist_handler/mnist_script.pt"
    )

    config = ModelArchiverConfig(
        model_name=model_name,
        serialized_file=mnist_scriptes_pt.as_posix(),
        model_file=None,
        handler="TorchScriptHandler",
        extra_files=None,
        runtime="LSP",
        export_path=work_dir,
        archive_format="default",
        force=True,
        version="1.0",
        requirements_file=None,
        config_file=None,
    )

    # Using ZIP_STORED instead of ZIP_DEFLATED reduces test runtime from 54 secs to 10 secs
    with patch(
        "model_archiver.model_packaging_utils.zipfile.ZipFile",
        lambda x, y, _: ZipFile(x, y, ZIP_STORED),
    ):
        model_archiver.generate_model_archive(config=config)

        assert mar_file_path.exists()

        print(mar_file_path)

        yield mar_file_path.as_posix()

    # Clean up files
    mar_file_path.unlink(missing_ok=True)


@pytest.fixture(scope="module", name="model_name_and_stdout")
def register_model(mar_file_path, model_store, torchserve):
    """
    Register the model in torchserve
    """

    print(os.environ["LD_LIBRARY_PATH"])

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

    assert test_utils.reg_resp.status_code == 200

    yield model_name, torchserve

    test_utils.unregister_model(model_name)


@pytest.mark.skipif(
    not (Path(ts.__file__).parent / "cpp" / "bin" / "model_worker_socket").exists(),
    reason="CPP backend not found",
)
def test_cpp_mnist(model_name_and_stdout):
    model_name, _ = model_name_and_stdout

    for n in range(10):
        data_file_mnist = (
            CURR_FILE_PATH.parents[1]
            / f"examples/image_classifier/mnist/test_data/{n}.png"
        ).as_posix()

        image_processing = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

        image = Image.open(data_file_mnist)
        image = image_processing(image)
        buffer = io.BytesIO()
        torch.save(image, buffer)

        url = f"http://localhost:8080/predictions/{model_name}/"

        buffer.seek(0)
        response = requests.post(url=url, data=buffer)

        assert response.status_code == 200

        assert torch.load(io.BytesIO(response.content)).argmax().item() == n
