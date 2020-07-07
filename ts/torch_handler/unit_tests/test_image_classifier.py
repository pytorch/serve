import pytest

@pytest.fixture
def model_setup():
    from .test_utils.mock_context import MockContext

    context = MockContext()
    with open('ts/torch_handler/unit_tests/models/tmp/images/kitten.jpg', 'rb') as fin:
        image_bytes = fin.read()
    return (context, image_bytes)

def test_initialize(model_setup):
    from ts.torch_handler.image_classifier import ImageClassifier
    model_context, _ = model_setup
    handler = ImageClassifier()
    handler.initialize(model_context)

    assert(True)
    return handler

def test_handle(model_setup):
    import torch

    model_context, image_bytes = model_setup
    handler = test_initialize(model_setup)
    test_data = [ {'data': image_bytes} ] * 2
    results = handler.handle(test_data, image_bytes)

    top_classes = [
        max(result.keys(), key=lambda k: result[k])
        for result in results
    ]

    assert(top_classes == ['tiger_cat'] * 2)
