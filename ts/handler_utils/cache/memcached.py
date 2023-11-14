import logging

from ts.handler_utils.cache.cache import Cache

logger = logging.getLogger(__name__)

try:
    import pymemcache

    _has_pymemcached = True
except ImportError:
    _has_pymemcached = False


class PyMemCached(Cache):
    def __init__(self, config=None):
        logger.info(f"Init pymemcache client with args {config}")

        if not _has_pymemcached:
            logger.error(f"Cannot import pymemcache, try pip install pymemcache.")
            return self._no_op_decorator
        self.client = pymemcache.client.base.Client(**config)

        try:
            result = self.client.set("some_key", "some_value")
        except ConnectionRefusedError:
            logger.error(
                f"Cannot connect to a memcached server, ensure a server is running on {config['server']}."
            )
