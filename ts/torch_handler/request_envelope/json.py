"""
Uses JSON formatted inputs/outputs, following the structure outlined if __name__ == '__main__':
https://www.tensorflow.org/tfx/serving/api_restmain()
"""
import json
from itertools import chain
from base64 import b64decode

from .base import BaseEnvelope

class JSONEnvelope(BaseEnvelope):
    """
    Implementation. Captures batches in JSON format, returns
    also in JSON format.
    """
    _lengths = []

    def parse_input(self, data):
        lengths, batch = self._batch_from_json(data)
        self._lengths = lengths
        return batch

    def format_output(self, data):
        return self._batch_to_json(data, self._lengths)

    def _batch_from_json(self, data_rows):
        """
        Joins the instances of a batch of JSON objects
        """
        mini_batches = [self._from_json(data_row) for data_row in data_rows]
        lengths = [len(mini_batch) for mini_batch in mini_batches]
        full_batch = list(chain.from_iterable(mini_batches))
        return lengths, full_batch

    def _from_json(self, data):
        """
        Extracts the data from the JSON object
        """
        rows = (data.get('data') or data.get('body') or data)['instances']
        if isinstance(rows[0], dict):
            for row_i, row in enumerate(rows):
                if list(row.keys()) == ['b64']:
                    rows[row_i] = b64decode(row['b64'])
                else:
                    for col, col_value in row.items():
                        if (isinstance(col_value, dict)
                                and list(col_value.keys()) == ['b64']):
                            row[col] = b64decode(col_value['b64'])
        return rows

    def _batch_to_json(self, batch, lengths):
        """
        Splits the batched output into mini-batches and returns JSON
        """
        outputs = []
        cursor = 0
        for length in lengths:
            cursor_end = cursor + length

            mini_batch = batch[cursor:cursor_end]
            outputs.append(self._to_json(mini_batch))

            cursor = cursor_end
        return outputs

    def _to_json(self, output):
        """
        Converts the output of the model back into compatible JSON
        """
        out_dict = {
            'predictions': output
        }
        return json.dumps(out_dict)
