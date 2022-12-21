"""
Module for image classification default handler
"""

from ts.cache.redis_cache import RedisCache
from ts.torch_handler.image_classifier import ImageClassifier


class CacheHandler(ImageClassifier):
    def __init__(self):
        self.handle = RedisCache()(self.handle)
