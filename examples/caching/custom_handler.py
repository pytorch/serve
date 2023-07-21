"""
Module for image classification default handler
"""

from ts.handler_utils.cache.utils import BackendCache
from ts.torch_handler.image_classifier import ImageClassifier


class CacheHandler(ImageClassifier):
    def __init__(self):
        super(CacheHandler, self).__init__()
        self.initialized = False

    def initialize(self, ctx):
        super().initialize(ctx)
        # self.handle = RedisCache(ctx)(self.handle)
        self.handle = BackendCache(ctx)(self.handle)
        self.initialized = True
