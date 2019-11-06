# Copyright 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#     http://www.apache.org/licenses/LICENSE-2.0
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

import mxnet as mx
import numpy as np
import sys
sys.path.append('../../..')

from mms.model_service.mxnet_model_service import MXNetBaseService, check_input_shape
from mms.utils.mxnet import image
from mxnet import ndarray as nd
from mxnet.gluon.nn import Dense, Activation, Conv2D, Conv2DTranspose, \
    BatchNorm, LeakyReLU, Flatten, HybridSequential, HybridBlock, Dropout


# Define Unet generator skip block
class UnetSkipUnit(HybridBlock):
    def __init__(self, inner_channels, outer_channels, inner_block=None, innermost=False, outermost=False,
                 use_dropout=False, use_bias=False):
        super(UnetSkipUnit, self).__init__()

        with self.name_scope():
            self.outermost = outermost
            en_conv = Conv2D(channels=inner_channels, kernel_size=4, strides=2, padding=1,
                             in_channels=outer_channels, use_bias=use_bias)
            en_relu = LeakyReLU(alpha=0.2)
            en_norm = BatchNorm(momentum=0.1, in_channels=inner_channels)
            de_relu = Activation(activation='relu')
            de_norm = BatchNorm(momentum=0.1, in_channels=outer_channels)

            if innermost:
                de_conv = Conv2DTranspose(channels=outer_channels, kernel_size=4, strides=2, padding=1,
                                          in_channels=inner_channels, use_bias=use_bias)
                encoder = [en_relu, en_conv]
                decoder = [de_relu, de_conv, de_norm]
                model = encoder + decoder
            elif outermost:
                de_conv = Conv2DTranspose(channels=outer_channels, kernel_size=4, strides=2, padding=1,
                                          in_channels=inner_channels * 2)
                encoder = [en_conv]
                decoder = [de_relu, de_conv, Activation(activation='tanh')]
                model = encoder + [inner_block] + decoder
            else:
                de_conv = Conv2DTranspose(channels=outer_channels, kernel_size=4, strides=2, padding=1,
                                          in_channels=inner_channels * 2, use_bias=use_bias)
                encoder = [en_relu, en_conv, en_norm]
                decoder = [de_relu, de_conv, de_norm]
                model = encoder + [inner_block] + decoder
            if use_dropout:
                model += [Dropout(rate=0.5)]

            self.model = HybridSequential()
            with self.model.name_scope():
                for block in model:
                    self.model.add(block)

    def hybrid_forward(self, F, x):
        if self.outermost:
            return self.model(x)
        else:
            return F.concat(self.model(x), x, dim=1)


# Define Unet generator
class UnetGenerator(HybridBlock):
    def __init__(self, in_channels, num_downs, ngf=64, use_dropout=True):
        super(UnetGenerator, self).__init__()

        # Build unet generator structure
        with self.name_scope():
            unet = UnetSkipUnit(ngf * 8, ngf * 8, innermost=True)
            for _ in range(num_downs - 5):
                unet = UnetSkipUnit(ngf * 8, ngf * 8, unet, use_dropout=use_dropout)
            unet = UnetSkipUnit(ngf * 8, ngf * 4, unet)
            unet = UnetSkipUnit(ngf * 4, ngf * 2, unet)
            unet = UnetSkipUnit(ngf * 2, ngf * 1, unet)
            unet = UnetSkipUnit(ngf, in_channels, unet, outermost=True)

            self.model = unet

    def hybrid_forward(self, F, x):
        return self.model(x)

class Pixel2pixelService(MXNetBaseService):

    def __init__(self, model_name, path):
        self.mx_model = UnetGenerator(in_channels=3, num_downs=8)
        self.mx_model.load_params('%s/%s.params' % (path, model_name), ctx=mx.cpu())

    def _preprocess(self, data):
        input_shape = self.signature['inputs'][0]['data_shape']
        height, width = input_shape[2:]
        img_arr = image.read(data[0])
        img_arr = image.resize(img_arr, width, height)
        img_arr = image.color_normalize(img_arr, nd.array([127.5]), nd.array([127.5]))
        img_arr = image.transform_shape(img_arr)
        return [img_arr]

    def _inference(self, data):
        check_input_shape(data, self.signature)
        return self.mx_model(*data)

    def _postprocess(self, data):
        img_arr = ((data[0] + 1.0) * 127.5).astype(np.uint8)
        return [image.write(img_arr)]




