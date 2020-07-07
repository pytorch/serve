import pytest
import sys

sys.path.append('ts/torch_handler/unit_tests/models/tmp')

@pytest.fixture
def model_context():
    from .test_utils.mock_context import MockContext

    return MockContext()

def test_initialize(model_context):
    from ts.torch_handler.base_handler import BaseHandler
    handler = BaseHandler()
    handler.initialize(model_context)

    assert(True)
    return handler

def test_single_handle(model_context):
    import torch
    handler = test_initialize(model_context)
    list_data = [[1.0, 2.0]]
    processed = handler.handle(list_data, model_context)

    assert(processed == [1])

def test_batch_handle(model_context):
    import torch
    handler = test_initialize(model_context)
    list_data = [[1.0, 2.0], [4.0, 3.0]]
    processed = handler.handle(list_data, model_context)

    assert(processed == [1, 0])
