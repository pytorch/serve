

import os
import sys
curr_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(curr_path + '/../../..')

import unittest
import mxnet as mx
import utils.mxnet.ndarray as ndarray

class TestMXNetNDArrayUtils(unittest.TestCase):
    def test_top_prob(self):
        labels = ['dummay' for _ in range(100)]
        data = mx.nd.random.uniform(0, 1, shape=(1, 100))
        top = 13
        output = ndarray.top_probability(data, labels, top=top)
        assert len(output) == top, "top_probability method failed."

    def runTest(self):
        self.test_top_prob()
