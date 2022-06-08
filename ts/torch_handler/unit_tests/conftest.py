import os

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


@pytest.fixture(autouse=True, scope="module")
def setup_directories():
    TEST_DIR = os.path.join("ts", "torch", "torch_handler", "unit_tests")

    os.system(f"mkdir -p {TEST_DIR}/models/tmp")
    yield
    os.system(f"rm -rf {TEST_DIR}/models/tmp")


@pytest.fixture()
def model_setup():
    context = MockContext(model_name="object_detector")
    with open("examples/image_segmenter/persons.jpg", "rb") as fin:
        image_bytes = fin.read()
    return (context, image_bytes)
