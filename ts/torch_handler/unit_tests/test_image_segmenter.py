import pytest
import sys

sys.path.append('ts/torch_handler/unit_tests/models/tmp')

@pytest.fixture
def model_setup():
    from .test_utils.mock_context import MockContext

    context = MockContext()
    with open('ts/torch_handler/unit_tests/models/tmp/persons.jpg', 'rb') as fin:
        image_bytes = fin.read()
    return (context, image_bytes)

def test_initialize(model_setup):
    from ts.torch_handler.image_segmenter import ImageSegmenter
    model_context, _ = model_setup
    handler = ImageSegmenter()
    handler.initialize(model_context)

    assert(True)
    return handler

def test_handle(model_setup):
    import torch

    model_context, image_bytes = model_setup
    handler = test_initialize(model_setup)
    test_data = [ {'data': image_bytes} ] * 2
    results = handler.handle(test_data, image_bytes)
    assert(len(results) == 2)
    assert(len(results[0]) == 21)
