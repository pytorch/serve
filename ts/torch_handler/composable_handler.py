import types

from ts.torch_handler.handler_utils import BaseInit, VisionInit, VisionPreproc
from ts.torch_handler.micro_batching import MicroBatchingHandler


class ComposableHandler(object):
    def __init__(self):
        mb_handle = MicroBatchingHandler(self)
        self.handle = mb_handle
        method_names = "initialize", "preprocess", "postprocess", "inference"

        self.initialize = types.MethodType(VisionInit(BaseInit()), self)
        self.preprocess = types.MethodType(VisionPreproc(), self)

    def _is_describe(self):
        return False

    def _is_explain(self):
        return False

    # for n, m in zip(method_names, methods):
    #     m = types.MethodType(m, self)
    #     setattr(self, n, m)
