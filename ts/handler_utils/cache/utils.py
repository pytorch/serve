from ts.handler_utils.cache.cache import Cache
from ts.handler_utils.cache.redis import RedisCache
from ts.utils.util import list_classes_from_module


def cache(func):
    def wrap_func(self, *args, **kwargs):
        global func
        if not self.cache_initialized:
            func = RedisCache(self.context)(func)
            self.cache_initialized = True
            result = func(self, *args, **kwargs)
        else:
            result = func(self, *args, **kwargs)

        return result

    return wrap_func


class BackendCache(Cache):
    def __init__(self, context=None):
        print("Init begin BC")
        ctx = context.model_yaml_config

        if "cache" not in ctx:
            assert "Cache config not specified"

        module = ctx["cache"]["module"]
        config = ctx["cache"]["config"]

        self.cache = self._get_cache_definition(module)

        self.cache(config)
        print("Init done  BC")

    def _get_cache_definition(self, module):
        print("Module is ", module)
        module = module.strip()
        print(module)

        if "redis" in module:
            return RedisCache
        cache_class_definitions = list_classes_from_module(module)
        print("Def is ", cache_class_definitions)
        if len(cache_class_definitions) != 1:
            raise ValueError(
                "Expected only one class in custom cache module {}".format(
                    cache_class_definitions
                )
            )

        cache_class = model_class_definitions[0]
        return cache_class
