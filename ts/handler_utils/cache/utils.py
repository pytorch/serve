import logging

from ts.handler_utils.cache.redis import RedisCache
from ts.utils.util import list_classes_from_module

logger = logging.getLogger(__name__)


def cache(func):
    def wrap_func(self, *args, **kwargs):
        result = func(self, *args, **kwargs)
        if self.model_yaml_config is not None and "cache" in self.model_yaml_config:
            module = self.model_yaml_config["cache"]["module"]
            config = self.model_yaml_config["cache"]["config"]
            cache_class = get_cache_definition(module)
            logger.info("Cache used is  %s", cache_class)
            self.handle = cache_class(config)(self.handle)
        return result

    return wrap_func


def get_cache_definition(module):
    logger.info("Module is ", module)
    module = module.strip()

    if "redis" in module:
        return RedisCache
    cache_class_definitions = list_classes_from_module(module)
    if len(cache_class_definitions) != 1:
        raise ValueError(
            "Expected only one class in custom cache module {}".format(
                cache_class_definitions
            )
        )

    cache_class = cache_class_definitions[0]
    return cache_class
