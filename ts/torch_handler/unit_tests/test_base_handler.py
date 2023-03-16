# pylint: disable=W0621
# Using the same name as global function is part of pytest
"""
Basic unit test for BaseHandler class.
Ensures it can load and execute an example model
"""

import pytest

from ts.torch_handler.base_handler import BaseHandler


@pytest.fixture()
def handler(base_model_context):
    handler = BaseHandler()
    handler.initialize(base_model_context)

    return handler


def test_single_handle(handler, base_model_context):
    list_data = [[1.0, 2.0]]
    processed = handler.handle(list_data, base_model_context)

    assert processed == [1]


def test_batch_handle(handler, base_model_context):
    list_data = [[1.0, 2.0], [4.0, 3.0]]
    processed = handler.handle(list_data, base_model_context)

    assert processed == [1, 0]
