# pylint: disable=W0621
# Using the same name as global function is part of pytest
"""
Basic unit test for BaseHandler class.
Ensures it can load and execute an example model
"""

import sys

from ts.torch_handler.request_envelope.body import BodyEnvelope
from ts.torch_handler.request_envelope.json import JSONEnvelope

sys.path.append("ts/torch_handler/unit_tests/models/tmp")


class TestEnvelopes:
    def test_json(self, handle_fn, model_context):
        test_data = [{"body": {"instances": [[1.0, 2.0]]}}]
        expected_result = ['{"predictions": [1]}']

        envelope = JSONEnvelope(handle_fn)
        results = envelope.handle(test_data, model_context)
        assert results == expected_result

    def test_json_batch(self, handle_fn, model_context):
        test_data = [{"body": {"instances": [[1.0, 2.0], [4.0, 3.0]]}}]
        expected_result = ['{"predictions": [1, 0]}']

        envelope = JSONEnvelope(handle_fn)
        results = envelope.handle(test_data, model_context)
        assert results == expected_result

    def test_json_double_batch(self, handle_fn, model_context):
        """
        More complex test case. Makes sure the model can
        mux several batches and return the demuxed results
        """
        test_data = [
            {"body": {"instances": [[1.0, 2.0]]}},
            {"body": {"instances": [[4.0, 3.0], [5.0, 6.0]]}},
        ]
        expected_result = ['{"predictions": [1]}', '{"predictions": [0, 1]}']

        envelope = JSONEnvelope(handle_fn)
        results = envelope.handle(test_data, model_context)
        print(results)
        assert results == expected_result

    def test_body(self, handle_fn, model_context):
        test_data = [{"body": [1.0, 2.0]}]
        expected_result = [1]

        envelope = BodyEnvelope(handle_fn)
        results = envelope.handle(test_data, model_context)
        assert results == expected_result

    def test_binary(self, model_context):
        test_data = [{"instances": [{"b64": "YQ=="}]}]

        envelope = JSONEnvelope(lambda x, y: [row.decode("utf-8") for row in x])
        results = envelope.handle(test_data, model_context)
        assert results == ['{"predictions": ["a"]}']
