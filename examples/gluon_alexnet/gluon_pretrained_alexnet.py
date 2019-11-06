# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#     http://www.apache.org/licenses/LICENSE-2.0
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

import mxnet
from gluon_base_service import GluonBaseService

"""
Gluon Pretrained Alexnet model
"""


class PretrainedAlexnetService(GluonBaseService):
    """
    Pretrained alexnet Service
    """
    def initialize(self, params):
        """
        Initialize the model
        :param params: This is the same as the Context object
        :return:
        """
        self.net = mxnet.gluon.model_zoo.vision.alexnet(pretrained=True)
        super(PretrainedAlexnetService, self).initialize(params)

    def postprocess(self, data):
        """
        Post process for the Gluon Alexnet model
        :param data:
        :return:
        """
        idx = data.topk(k=5)[0]
        return [[{'class': (self.labels[int(i.asscalar())]).split()[1], 'probability':
                float(data[0, int(i.asscalar())].asscalar())} for i in idx]]


svc = PretrainedAlexnetService()


def pretrained_gluon_alexnet(data, context):
    """
    This is the handler that needs to be registerd in the model-archive.
    :param data:
    :param context:
    :return:
    """
    res = None
    if not svc.initialized:
        svc.initialize(context)

    if data is not None:
        res = svc.predict(data)

    return res
