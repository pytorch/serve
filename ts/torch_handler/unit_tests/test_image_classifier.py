# pylint: disable=W0621
# Using the same name as global function is part of pytest
"""
Basic unit test for ImageClassifier class.
Ensures it can load and execute an example model
"""

import os

import pytest

from ts.torch_handler.image_classifier import ImageClassifier

from .test_utils.mock_context import MockContext


# sys.path.append("ts/torch_handler/unit_tests/models/tmp")
class TestImageClassifier:
    @pytest.fixture()
    def model_setup(self):
        TEST_DIR = os.path.join("ts", "torch_handler", "unit_tests", "models", "tmp")
        os.system(f"python {TEST_DIR}/models/base_model.py")
        os.system(
            f"wget -nc -q -O {TEST_DIR}/models/tmp/model.pt https://download.pytorch.org/models/resnet152-b121ed2d.pth"
        )
        os.system(
            f"cp -r examples/image_classifier/resnet_152_batch/* {TEST_DIR}/models/tmp"
        )
        os.system(f"cp examples/image_classifier/kitten.jpg {TEST_DIR}/models/tmp")

        context = MockContext(model_name="image_classifier")
        with open("examples/image_classifier/kitten.jpg", "rb") as fin:
            image_bytes = fin.read()

        handler = ImageClassifier()
        handler.initialize(context)
        return (context, image_bytes, handler)

    def test_handle(self, model_setup):
        context, image_bytes, handler = model_setup
        test_data = [{"data": image_bytes}]
        results = handler.handle(test_data, context)
        assert len(results) == 1

    # def test_handle_explain(self, model_setup):
    #     context, image_bytes, handler = model_setup
    #     context.explain = True
    #     test_data = [{"data": image_bytes, "target": 0}]
    #     results = handler.handle(test_data, context)
    #     assert len(results) == 1
