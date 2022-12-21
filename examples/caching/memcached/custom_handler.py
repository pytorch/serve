"""
Module for image classification default handler
"""

from ts.cache.memcached import MemCached
from ts.torch_handler.image_classifier import ImageClassifier


class CacheHandler(ImageClassifier):
    def __init__(self):
        self.handle = MemCached()(self.handle)
