# pylint: disable=W0621
# Using the same name as global function is part of pytest
"""
Basic unit test for ImageSegmenter class.
Ensures it can load and execute an example model
"""

import os
import sys

import pytest

from ts.torch_handler.image_segmenter import ImageSegmenter

from .test_utils.mock_context import MockContext

sys.path.append("ts/torch_handler/unit_tests/models/tmp")


@pytest.fixture()
def model_setup():
    context = MockContext(model_name="image_segmenter")
    persons_path = os.path.join("examples", "image_segmenter", "persons.jpg")
    with open(persons_path, "rb") as fin:
        image_bytes = fin.read()
    return (context, image_bytes)


class TestImageSegmenter:
    def test_initialize(self, model_setup):
        TEST_DIR = "./ts/torch_handler/unit_tests"
        os.system(
            f"wget -nc -q -O {TEST_DIR}/models/tmp/model.pt https://download.pytorch.org/models/fcn_resnet101_coco-7ecb50ca.pth"
        )
        os.system(f"cp -r examples/image_segmenter/fcn/* {TEST_DIR}/models/tmp")
        model_context, _ = model_setup
        handler = ImageSegmenter()
        handler.initialize(model_context)

        assert True
        return handler

    def test_handle(self, model_setup):
        context, image_bytes = model_setup
        handler = self.test_initialize(model_setup)
        test_data = [{"data": image_bytes}]
        results = handler.handle(test_data, context)
        assert len(results) == 1
