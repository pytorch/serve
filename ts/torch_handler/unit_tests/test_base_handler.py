# pylint: disable=W0621
# Using the same name as global function is part of pytest
"""
Basic unit test for BaseHandler class.
Ensures it can load and execute an example model
"""

import sys

from ts.torch_handler.base_handler import BaseHandler

sys.path.append("ts/torch_handler/unit_tests/models/tmp")


def test_initialize(model_context):
    handler = BaseHandler()
    handler.initialize(model_context)

    assert True
    return handler


def test_single_handle(model_context):
    handler = test_initialize(model_context)
    list_data = [[1.0, 2.0]]
    processed = handler.handle(list_data, model_context)

    assert processed == [1]


def test_batch_handle(model_context):
    handler = test_initialize(model_context)
    list_data = [[1.0, 2.0], [4.0, 3.0]]
    processed = handler.handle(list_data, model_context)

    assert processed == [1, 0]
