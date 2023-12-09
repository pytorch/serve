# pylint: disable=W0621
# Using the same name as global function is part of pytest
"""
Basic unit test for ImageClassifier class.
Ensures it can load and execute an example model
"""
import sys
from pathlib import Path

import pytest
from torchvision.models.resnet import ResNet18_Weights

from ts.torch_handler.image_classifier import ImageClassifier

from .test_utils.mock_context import MockContext
from .test_utils.model_dir import copy_files, download_model

REPO_DIR = Path(__file__).parents[3]


@pytest.fixture(scope="module")
def image_bytes():
    with open(
        REPO_DIR.joinpath(
            "examples/image_classifier/resnet_152_batch/images/kitten.jpg"
        ).as_posix(),
        "rb",
    ) as fin:
        image_bytes = fin.read()
    yield image_bytes


@pytest.fixture(scope="module")
def model_name():
    return "image_classifier"


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory, model_name):
    model_dir = tmp_path_factory.mktemp("image_classifier_model_dir")

    src_dir = REPO_DIR.joinpath("examples/image_classifier/resnet_18/")

    model_url = ResNet18_Weights.DEFAULT.url

    download_model(model_url, model_dir)

    files = {
        "model.py": model_name + ".py",
        "index_to_name.json": "index_to_name.json",
    }

    copy_files(src_dir, model_dir, files)

    sys.path.append(model_dir.as_posix())
    yield model_dir
    sys.path.pop()


@pytest.fixture(scope="module")
def context(model_dir, model_name):

    context = MockContext(
        model_name="mnist",
        model_dir=model_dir.as_posix(),
        model_file=model_name + ".py",
    )
    yield context


@pytest.fixture(scope="module")
def handler(context):
    handler = ImageClassifier()
    handler.initialize(context)
    return handler


def test_handle(context, image_bytes, handler):
    test_data = [{"data": image_bytes}] * 2
    results = handler.handle(test_data, context)
    assert len(results) == 2
    assert "tiger_cat" in results[0]


def test_handle_explain(context, image_bytes, handler):
    context.explain = True
    test_data = [{"data": image_bytes, "target": 0}] * 2
    results = handler.handle(test_data, context)
    assert len(results) == 2
    assert results[0]
