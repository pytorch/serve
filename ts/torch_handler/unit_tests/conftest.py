import pytest

from ts.torch_handler.base_handler import BaseHandler

from .test_utils.mock_context import MockContext


@pytest.fixture()
def model_context():
    return MockContext()


@pytest.fixture()
def handle_fn():
    ctx = MockContext()
    handler = BaseHandler()
    handler.initialize(ctx)
    return handler.handle


@pytest.fixture()
def model_setup():
    context = MockContext(model_name="object_detector")
    with open("examples/image_segmenter/persons.jpg", "rb") as fin:
        image_bytes = fin.read()
    return (context, image_bytes)
