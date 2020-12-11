# pylint: disable=W0621
# Using the same name as global function is part of pytest
"""
Basic unit test for ImageClassifier class.
Ensures it can load and execute an example model
"""

import sys
import pytest
from ts.torch_handler.image_classifier import ImageClassifier
from .test_utils.mock_context import MockContext

sys.path.append('ts/torch_handler/unit_tests/models/tmp')

@pytest.fixture()
def model_setup():
    context = MockContext(model_name="mnist")
    with open('ts/torch_handler/unit_tests/models/tmp/images/kitten.jpg', 'rb') as fin:
        image_bytes = fin.read()
    return (context, image_bytes)

def test_initialize(model_setup):
    model_context, _ = model_setup
    handler = ImageClassifier()
    handler.initialize(model_context)

    assert(True)
    return handler

def test_handle(model_setup):
    context, image_bytes = model_setup
    handler = test_initialize(model_setup)
    test_data = [{'data': image_bytes}] * 2
    results = handler.handle(test_data, context)
    assert(len(results) == 2)
    assert('tiger_cat' in results[0])

def test_handle_explain(model_setup):
    context, image_bytes = model_setup
    context.explain = True
    handler = test_initialize(model_setup)
    test_data = [{'data': image_bytes, 'target': 0}] * 2
    results = handler.handle(test_data, context)
    assert(len(results) == 2)
    assert(results[0])
