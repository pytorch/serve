# Copyright 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#     http://www.apache.org/licenses/LICENSE-2.0
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

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
