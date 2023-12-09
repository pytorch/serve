# pylint: disable=W0621
# Using the same name as global function is part of pytest
"""
Basic unit test for ImageSegmenter class.
Ensures it can load and execute an example model
"""

import sys
from pathlib import Path

import pytest

from ts.torch_handler.image_segmenter import ImageSegmenter

from .test_utils.mock_context import MockContext
from .test_utils.model_dir import copy_files, download_model

REPO_DIR = Path(__file__).parents[3]


@pytest.fixture()
def image_bytes():
    with open(REPO_DIR.joinpath("examples/image_segmenter/persons.jpg"), "rb") as fin:
        image_bytes = fin.read()
    return image_bytes


@pytest.fixture()
def model_name():
    return "image_segmenter"


@pytest.fixture()
def model_dir(tmp_path_factory, model_name):
    model_dir = tmp_path_factory.mktemp("image_segmenter_model_dir")

    src_dir = REPO_DIR.joinpath("examples/image_segmenter/fcn/")

    model_url = "https://download.pytorch.org/models/fcn_resnet101_coco-7ecb50ca.pth"

    download_model(model_url, model_dir)

    files = {
        "model.py": model_name + ".py",
        "intermediate_layer_getter.py": "intermediate_layer_getter.py",
        "fcn.py": "fcn.py",
    }

    copy_files(src_dir, model_dir, files)

    sys.path.append(model_dir.as_posix())
    yield model_dir
    sys.path.pop()


@pytest.fixture()
def context(model_dir, model_name):

    context = MockContext(
        model_name="mnist",
        model_dir=model_dir.as_posix(),
        model_file=model_name + ".py",
    )
    yield context


@pytest.fixture()
def handler(context):
    handler = ImageSegmenter()
    handler.initialize(context)

    return handler


def test_handle(handler, context, image_bytes):
    test_data = [{"data": image_bytes}] * 2
    results = handler.handle(test_data, context)

    assert len(results) == 2
    assert len(results[0]) == 224
