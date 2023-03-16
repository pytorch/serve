# pylint: disable=W0621
# Using the same name as global function is part of pytest
"""
Basic unit test for ImageClassifier class.
Ensures it can load and execute an example model
"""

import io
import shutil
import sys
from pathlib import Path

import pytest
import torchvision.transforms as transforms
from PIL import Image

from examples.image_classifier.mnist.mnist_handler import (
    MNISTDigitClassifier as MNISTClassifier,
)
from ts.torch_handler.request_envelope import kserve, kservev2

from .test_utils.mock_context import MockContext

REPO_DIR = Path(__file__).parents[3]


image_processing = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)


@pytest.fixture(scope="module")
def image_bytes():

    with open(
        REPO_DIR.joinpath("examples/image_classifier/mnist/test_data/0.png"), "rb"
    ) as fin:
        image_bytes = fin.read()
    return image_bytes


@pytest.fixture(scope="module")
def context(tmp_path_factory):
    model_dir = tmp_path_factory.mktemp("model_dir")

    shutil.copytree(
        REPO_DIR.joinpath("examples/image_classifier/mnist/"),
        model_dir,
        dirs_exist_ok=True,
    )

    context = MockContext(
        model_name="mnist",
        model_dir=model_dir.as_posix(),
        model_file="mnist.py",
        model_pt_file="mnist_cnn.pt",
    )

    sys.path.append(model_dir.as_posix())
    yield context
    sys.path.pop()


@pytest.fixture(scope="module")
def handler(context):
    handler = MNISTClassifier()
    handler.initialize(context)
    return handler


@pytest.fixture()
def envelope_kf(context):
    handler = MNISTClassifier()
    handler.initialize(context)
    envelope = kserve.KServeEnvelope(handler.handle)
    return envelope


@pytest.fixture()
def envelope_kfv2(context):
    handler = MNISTClassifier()
    handler.initialize(context)
    envelope = kservev2.KServev2Envelope(handler.handle)
    return envelope


def test_handle(handler, context, image_bytes):
    test_data = [{"data": image_bytes}]
    # testing for predict API
    results = handler.handle(test_data, context)
    assert results[0] in range(0, 9)


def test_handle_kf(envelope_kf, context, image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    image_list = image_processing(image).tolist()
    test_data = {"body": {"instances": [{"data": image_list}]}}

    # testing for predict API
    results = envelope_kf.handle([test_data], context)
    assert results[0]["predictions"][0] in range(0, 9)


def test_handle_kfv2(envelope_kfv2, context, image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    image_list = image_processing(image).tolist()
    test_data = {
        "body": {
            "id": "test-id",
            "inputs": [
                {
                    "data": image_list,
                    "datatype": "FP32",
                    "name": "test-input",
                    "shape": [1, 28, 28],
                }
            ],
        }
    }

    # testing for v2predict API
    results = envelope_kfv2.handle([test_data], context)
    print(results)
    assert results[0]["outputs"][0]["data"][0] in range(0, 9)
