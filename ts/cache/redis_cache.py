import logging

from ts.cache.cache import Cache

try:
    import redis

    _has_redis = True
except ImportError:
    _has_redis = False


class RedisCache(Cache):
    def __init__(self, host="localhost", port=6379, db=0):

        if not _has_redis:
            logging.error(f"Cannot import redis, try pip install redis.")
            return self._no_op_decorator
        self.client = redis.Redis(host=host, port=port, db=db)
        try:
            self.client.ping()
        except redis.exceptions.ConnectionError:
            logging.error(
                f"Cannot connect to a Redis server, ensure a server is running on {host}:{port}."
            )
