# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#     http://www.apache.org/licenses/LICENSE-2.0
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

from mxnet.gluon import nn
from mxnet.gluon.block import HybridBlock
from gluon_base_service import GluonBaseService

"""
MMS examples for loading Gluon Hybrid models
"""


class GluonHybridAlexNet(HybridBlock):
    """
    Hybrid Block gluon model
    """
    def __init__(self, classes=1000, **kwargs):
        """
        This is the network definition of Gluon Hybrid Alexnet
        :param classes:
        :param kwargs:
        """
        super(GluonHybridAlexNet, self).__init__(**kwargs)
        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')
            with self.features.name_scope():
                self.features.add(nn.Conv2D(64, kernel_size=11, strides=4,
                                            padding=2, activation='relu'))
                self.features.add(nn.MaxPool2D(pool_size=3, strides=2))
                self.features.add(nn.Conv2D(192, kernel_size=5, padding=2,
                                            activation='relu'))
                self.features.add(nn.MaxPool2D(pool_size=3, strides=2))
                self.features.add(nn.Conv2D(384, kernel_size=3, padding=1,
                                            activation='relu'))
                self.features.add(nn.Conv2D(256, kernel_size=3, padding=1,
                                            activation='relu'))
                self.features.add(nn.Conv2D(256, kernel_size=3, padding=1,
                                            activation='relu'))
                self.features.add(nn.MaxPool2D(pool_size=3, strides=2))
                self.features.add(nn.Flatten())
                self.features.add(nn.Dense(4096, activation='relu'))
                self.features.add(nn.Dropout(0.5))
                self.features.add(nn.Dense(4096, activation='relu'))
                self.features.add(nn.Dropout(0.5))

            self.output = nn.Dense(classes)

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.output(x)
        return x


class HybridAlexnetService(GluonBaseService):
    """
    Gluon alexnet Service
    """
    def initialize(self, params):
        self.net = GluonHybridAlexNet()
        self.param_filename = "alexnet.params"
        super(HybridAlexnetService, self).initialize(params)
        self.net.hybridize()

    def postprocess(self, data):
        idx = data.topk(k=5)[0]
        return [[{'class': (self.labels[int(i.asscalar())]).split()[1], 'probability':
                 float(data[0, int(i.asscalar())].asscalar())} for i in idx]]


svc = HybridAlexnetService()


def hybrid_gluon_alexnet_inf(data, context):
    res = None
    if not svc.initialized:
        svc.initialize(context)

    if data is not None:
        res = svc.predict(data)

    return res
