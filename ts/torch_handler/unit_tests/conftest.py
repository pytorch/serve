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


@pytest.fixture(autouse=True, scope="class")
def setup_directories():
    TEST_DIR = os.path.join(
        "ts", "torch", "torch_handler", "unit_tests", "models", "tmp"
    )

    os.system(f"mkdir -p {TEST_DIR}")
    yield
    os.system(f"rm -rf {TEST_DIR}")


# Function for create, download or move model


@pytest.fixture()
def model_setup():
    context = MockContext(model_name="object_detector")
    persons_path = os.path.join("examples", "image_segmenter", "persons.jpg")
    with open(persons_path, "rb") as fin:
        image_bytes = fin.read()
    return (context, image_bytes)
