

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
