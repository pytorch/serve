import logging

from ts.handler_utils.cache.cache import Cache

logger = logging.getLogger(__name__)

try:
    import redis

    _has_redis = True
except ImportError:
    _has_redis = False


class RedisCache(Cache):
    def __init__(self, config=None):
        logger.info(f"Init Redis client with args {config}")

        if not _has_redis:
            logger.error(f"Cannot import redis, try pip install redis.")
            return self._no_op_decorator
        self.client = redis.Redis(**config)
        try:
            self.client.ping()
        except redis.exceptions.ConnectionError:
            logger.error(
                f"Cannot connect to a Redis server, ensure a server is running on {config['host']}:{config['port']}."
            )
