# pylint: disable=W0621
# Using the same name as global function is part of pytest
"""
Basic unit test for ImageSegmenter class.
Ensures it can load and execute an example model
"""

import sys

from ts.torch_handler.image_segmenter import ImageSegmenter

sys.path.append("ts/torch_handler/unit_tests/models/tmp")


class TestImageSegmenter:
    def test_initialize(self, model_setup):
        model_context, _ = model_setup
        handler = ImageSegmenter()
        handler.initialize(model_context)

        assert True
        return handler

    def test_handle(self, model_setup):
        context, image_bytes = model_setup
        handler = self.test_initialize(model_setup)
        test_data = [{"data": image_bytes}] * 2
        results = handler.handle(test_data, context)
        assert len(results) == 2
        assert len(results[0]) == 224
