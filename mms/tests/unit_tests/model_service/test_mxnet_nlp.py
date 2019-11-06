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
import utils.mxnet.nlp as nlp

from random import randint

class TestMXNetNLPUtils(unittest.TestCase):
    def test_encode_sentence(self):
        vocab = {}
        sentence = []
        for i in range(100):
            vocab['word%d' % (i)] = i
        sen_vec = [0, 56, 8, 10]
        for i in sen_vec:
            sentence.append('word%d' % (i))
        res1, out1 = nlp.encode_sentences([sentence], vocab)
        assert res1[0] == sen_vec, "encode_sentence method failed. " \
                                   "Result vector invalid."
        assert len(out1) == len(vocab), "encode_sentence method failed. " \
                                        "Generated vocab incorrect."

        res2, out2 = nlp.encode_sentences([sentence])
        assert res2[0] == [i for i in range(len(sentence))], \
            "encode_sentence method failed. Result vector invalid."
        assert len(out2) == len(sentence) + 1, "encode_sentence method failed. " \
                                               "Generated vocab incorrect."

    def test_pad_sentence(self):
        buckets = [10, 20, 30, 40, 50, 60]
        for _ in range(5):
            sent_length = randint(1, 60)
            sentence = [i for i in range(sent_length)]
            databatch = nlp.pad_sentence(sentence, buckets)
            assert databatch.data[0].shape[1] in buckets, "pad_sentence failed. Padded sentence has length %d." \
                                                          % (databatch.data[0].shape[1])


