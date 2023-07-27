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
        # host, port, db = "localhost", 6379, 0

        # if not isinstance(ctx, type(None)):
        #    print("Reading from yaml")
        #    host = ctx.model_yaml_config["cache"]["host"]
        #    port = ctx.model_yaml_config["cache"]["port"]
        #    db = ctx.model_yaml_config["cache"]["db"]

        args = {}
        for k, v in config.items():
            args[k] = v

        logger.info("$$$$$$$$$$$$$$$$$$$$$$$$$$ Init Redis")
        logger.info("args", args)

        if not _has_redis:
            logger.error(f"Cannot import redis, try pip install redis.")
            return self._no_op_decorator
        # self.client = redis.Redis(host=host, port=port, db=db)
        self.client = redis.Redis(**args)
        try:
            self.client.ping()
        except redis.exceptions.ConnectionError:
            logger.error(
                f"Cannot connect to a Redis server, ensure a server is running on {args['host']}:{args['port']}."
            )
