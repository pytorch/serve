# pylint: disable=W0621
# Using the same name as global function is part of pytest
"""
Basic unit test for ObjectDetector class.
Ensures it can load and execute an example model
"""

import os
import sys

import pytest

from ts.torch_handler.object_detector import ObjectDetector

from .test_utils.mock_context import MockContext

sys.path.append("ts/torch_handler/unit_tests/models/tmp")


class TestObjectDetector:
    @pytest.fixture()
    def model_setup(self):
        TEST_DIR = "./ts/torch_handler/unit_tests"

        os.system(
            f"wget -nc -q -O {TEST_DIR}/models/tmp/model.pt https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth"
        )
        os.system(f"cp -r examples/object_detector/fast-rcnn/* {TEST_DIR}/models/tmp")
        context = MockContext(model_name="object_detector")
        with open("./ts/torch_handler/unit_tests/models/tmp/persons.jpg", "rb") as fin:
            image_bytes = fin.read()
        return (context, image_bytes)

    def test_initialize(self, model_setup):
        model_context, _ = model_setup
        handler = ObjectDetector()
        handler.initialize(model_context)

        assert True
        return handler

    def test_handle(self, model_setup):
        context, image_bytes = model_setup
        handler = self.test_initialize(model_setup)
        test_data = [{"data": image_bytes}]
        results = handler.handle(test_data, context)
        assert len(results) == 1
