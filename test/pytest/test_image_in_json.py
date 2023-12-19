import platform
import shutil
from pathlib import Path

import pytest
import requests
import test_utils
from model_archiver import ModelArchiverConfig

CURR_FILE_PATH = Path(__file__).parent
REPO_ROOT_DIR = CURR_FILE_PATH.parent.parent

MODEL_PY = """
import torch
import torch.nn as nn

class Foo(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
"""

HANDLER_PY = """
import os
import torch
from ts.handler_utils.utils import send_intermediate_predict_response
from ts.torch_handler.base_handler import BaseHandler

class FooHandler(BaseHandler):
    def initialize(self, ctx):
        torch.set_default_device("cpu")
        self.context = ctx
        super().initialize(ctx)

    def preprocess(self, data):
        print(f'{data[0]["params1"]=}')
        print(f'{data[0]["params2"]=}')
        print(f'{data[0]["params3"]=}')
        import PIL.Image as Image
        import io
        import base64

        img = Image.open(io.BytesIO(base64.b64decode(data[0]['image_data'])))

        return img.size

    def inference(self, x):
        return x

    def postprocess(self, x):
        return [x]
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

    model_py_file = work_dir / "model.py"
    model_py_file.write_text(MODEL_PY)

    handler_py_file = work_dir / "handler.py"
    handler_py_file.write_text(HANDLER_PY)

    config = ModelArchiverConfig(
        model_name=model_name,
        version="1.0",
        serialized_file=None,
        model_file=model_py_file.as_posix(),
        handler=handler_py_file.as_posix(),
        extra_files=None,
        export_path=work_dir,
        requirements_file=None,
        runtime="python",
        force=False,
        archive_format="default",
        config_file=None,
    )

    model_archiver.generate_model_archive(config=config)

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

    assert test_utils.reg_resp.status_code == 200

    yield model_name

    test_utils.unregister_model(model_name)


@pytest.mark.skipif(
    platform.system() != "Linux", reason="Skipping test on non-Linux system"
)
def test_tp_inference(model_name):
    """
    Full circle test with torchserve
    """

    import base64

    image_file = REPO_ROOT_DIR / "examples/image_classifier/kitten.jpg"

    with open(image_file, "rb") as f:
        image_bytes = f.read()

    import io

    import PIL.Image as Image

    img = Image.open(io.BytesIO(image_bytes))

    encoded = base64.b64encode(image_bytes)

    json_data = {
        "params1": "meta_data1",
        "params2": "meta_data2",
        "params3": "meta_data3",
        "image_data": encoded,
    }

    response = requests.post(
        url=f"http://localhost:8080/predictions/{model_name}", data=json_data
    )

    assert response.status_code == 200

    print(f"{response.text=}")
