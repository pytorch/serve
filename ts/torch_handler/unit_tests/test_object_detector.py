# pylint: disable=W0621
# Using the same name as global function is part of pytest
"""
Basic unit test for ObjectDetector class.
Ensures it can load and execute an example model
"""

import sys
import pytest
from ts.torch_handler.object_detector import ObjectDetector
from .test_utils.mock_context import MockContext

sys.path.append('ts/torch_handler/unit_tests/models/tmp')

@pytest.fixture()
def model_setup():
    context = MockContext()
    with open('ts/torch_handler/unit_tests/models/tmp/persons.jpg', 'rb') as fin:
        image_bytes = fin.read()
    return (context, image_bytes)

def test_initialize(model_setup):
    model_context, _ = model_setup
    handler = ObjectDetector()
    handler.initialize(model_context)

    assert(True)
    return handler

def test_handle(model_setup):
    _, image_bytes = model_setup
    handler = test_initialize(model_setup)
    test_data = [{'data': image_bytes}] * 2
    results = handler.handle(test_data, image_bytes)
    assert(len(results) == 2)
    assert('bench' in results[0])
