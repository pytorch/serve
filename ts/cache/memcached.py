import logging

from ts.cache.cache import Cache

try:
    import pymemcache

    _has_memcached = True
except ImportError:
    _has_memcached = False


class MemCached(Cache):
    def __init__(self, host="localhost", port=11211):

        if not _has_memcached:
            logging.error(f"Cannot import pymemcache, try pip install pymemcache.")
            return self._no_op_decorator
        self.client = pymemcache.client.base.Client((host, port))
        try:
            self.client._connect()
        except ConnectionRefusedError:
            logging.error(
                f"Cannot connect to a memcached server, ensure a server is running on {host}:{port}."
            )
