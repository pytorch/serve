# pylint: disable=W0621
# Using the same name as global function is part of pytest
"""
Basic unit test for ObjectDetector class.
Ensures it can load and execute an example model
"""

import platform
import sys
from pathlib import Path

import pytest

from ts.torch_handler.object_detector import ObjectDetector

from .test_utils.mock_context import MockContext
from .test_utils.model_dir import copy_files, download_model

REPO_DIR = Path(__file__).parents[3]


@pytest.fixture(scope="module")
def image_bytes():
    with open(REPO_DIR.joinpath("examples/image_segmenter/persons.jpg"), "rb") as fin:
        image_bytes = fin.read()
    return image_bytes


@pytest.fixture(scope="module")
def model_name():
    return "object_detector"


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory, model_name):
    model_dir = tmp_path_factory.mktemp("model_dir")

    src_dir = REPO_DIR.joinpath("examples/object_detector/fast-rcnn/")

    model_url = (
        "https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth"
    )

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
    handler = ObjectDetector()
    handler.initialize(context)

    return handler


@pytest.mark.skipif(
    platform.machine() == "aarch64", reason="Test skipped on aarch64 architecture"
)
def test_handle(handler, context, image_bytes):
    test_data = [{"data": image_bytes}] * 2
    results = handler.handle(test_data, context)
    assert len(results) == 2
    assert any("bench" in d for d in results[0])
