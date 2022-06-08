# pylint: disable=W0621
# Using the same name as global function is part of pytest
"""
Basic unit test for BaseHandler class.
Ensures it can load and execute an example model
"""

import os
import sys

from ts.torch_handler.base_handler import BaseHandler

sys.path.append("ts/torch_handler/unit_tests/models/tmp")


class TestBaseHandler:
    def test_initialize(self, model_context):
        TEST_DIR = "./ts/torch_handler/unit_tests"

        os.system(f"python {TEST_DIR}/models/base_model.py")
        os.system(f"mv base_model.pt {TEST_DIR}/models/tmp/model.pt")
        os.system(f"cp {TEST_DIR}/models/base_model.py {TEST_DIR}/models/tmp/model.py")
        handler = BaseHandler()
        handler.initialize(model_context)

        assert True
        return handler

    def test_single_handle(self, model_context):
        handler = self.test_initialize(model_context)
        list_data = [[1.0, 2.0]]
        processed = handler.handle(list_data, model_context)

        assert processed == [1]

    def test_batch_handle(self, model_context):
        handler = self.test_initialize(model_context)
        list_data = [[1.0, 2.0], [4.0, 3.0]]
        processed = handler.handle(list_data, model_context)

        assert processed == [1, 0]
