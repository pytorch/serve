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

import PIL
import unittest
import numpy as np
import mxnet as mx
import mms.utils.mxnet.image as image
from io import BytesIO

class TestMXNetImageUtils(unittest.TestCase):
    def _write_image(self, img_arr, flag=1):
        img_arr = mx.nd.transpose(img_arr, (1, 2, 0))
        mode = 'RGB' if flag == 1 else 'L'
        if flag == 0:
            img_arr = mx.nd.reshape(img_arr, shape=(img_arr.shape[0], img_arr.shape[1]))
        img_arr = img_arr.astype(np.uint8).asnumpy()
        image = PIL.Image.fromarray(img_arr, mode)
        output = BytesIO()
        image.save(output, format='jpeg')
        return output.getvalue()

    def test_transform_shape(self):
        input1 = mx.nd.random.uniform(0, 255, shape=(32, 32, 3))
        output1 = image.transform_shape(input1)
        assert output1.shape == (1, 3, 32, 32), "transform_shape method fail. Got %s shape." % (str(output1.shape))

        input2 = mx.nd.random.uniform(0, 255, shape=(28, 28, 3))
        output2 = image.transform_shape(input2, dim_order='NHWC')
        assert output2.shape == (1, 28, 28, 3), "transform_shape method fail. Got %s shape." % (str(output2.shape))

    def test_read(self):
        input1 = mx.nd.random.uniform(0, 255, shape=(3, 256, 256))
        input_buf1 = self._write_image(input1)
        output1 = image.read(input_buf1)
        assert output1.shape == (256, 256, 3), "Read method failed. Got %s shape." % (str(output1.shape))

        input2 = mx.nd.random.uniform(0, 255, shape=(1, 128, 128))
        input_buf2 = self._write_image(input2, flag=0)
        output2 = image.read(input_buf2, flag=0)
        assert output2.shape == (128, 128, 1), "Read method failed. Got %s shape." % (str(output2.shape))

    def test_write(self):
        input1 = mx.nd.random.uniform(0, 255, shape=(3, 256, 256))
        output1 = image.write(input1)
        assert isinstance(output1, str), "Write method failed. Output is not a string."

        input2 = mx.nd.random.uniform(0, 255, shape=(256, 256, 1))
        output2 = image.write(input2, flag=0, dim_order='HWC')
        assert isinstance(output2, str), "Write method failed. Output is not a string."

    def test_resize(self):
        input1 = mx.nd.random.uniform(0, 255, shape=(245, 156, 3))
        output1 = image.resize(input1, 128, 256)
        assert output1.shape == (256, 128, 3), "Resize method failed. Got %s shape." % (str(output1.shape))

    def test_fix_crop(self):
        input1 = mx.nd.random.uniform(0, 255, shape=(100, 100, 3))
        output1 = image.fixed_crop(input1, 10, 20, 50, 70)
        assert output1.shape == (70, 50, 3), "Resize method failed. Got %s shape." % (str(output1.shape))

    def test_color_normalize(self):
        input1 = mx.nd.random.uniform(0, 255, shape=(1, 10, 10))
        output1 = image.color_normalize(input1, 127.5, 127.5).asnumpy()
        assert (output1 >= -1.0).all() and (output1 <= 1.0).all(), "color_normalize method failed."

    def runTest(self):
        self.test_transform_shape()
        self.test_read()
        self.test_write()
        self.test_resize()
        self.test_fix_crop()
        self.test_color_normalize()