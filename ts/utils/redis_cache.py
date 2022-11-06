import logging
import pickle
from functools import wraps

try:
    import redis

    _has_redis = True
except ImportError:
    _has_redis = False

from ts.context import Context


def _make_key(args, kwds):
    key = args
    if kwds:
        key += (object(),)
        for item in kwds.items():
            key += item
    return pickle.dumps(key)


def _no_op_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwds):
        return func(*args, **kwds)

    return wrapper


def handler_cache(host, port, db, maxsize=128):
    """Decorator for handler's handle() method that cache input/output to a Redis database.

    A typical usage would be:

    class SomeHandler(BaseHandler):
        def __init__(self):
            ...
            self.handle = handler_cache(host='localhost', port=6379, db=0, maxsize=128)(self.handle)

    The user should ensure that both the input and the output can be pickled.
    """
    if not _has_redis:
        logging.error(f"Cannot import redis, try pip install redis.")
        return _no_op_decorator
    r = redis.Redis(host=host, port=port, db=db)
    try:
        r.ping()
    except redis.exceptions.ConnectionError:
        logging.error(
            f"Cannot connect to a Redis server, ensure a server is running on {host}:{port}."
        )
        return _no_op_decorator

    def decorating_function(func):
        @wraps(func)
        def wrapper(*args, **kwds):
            # Removing Context objects from key hashing
            key = _make_key(
                args=[arg for arg in args if not isinstance(arg, Context)],
                kwds={k: v for (k, v) in kwds.items() if not isinstance(v, Context)},
            )
            value_str = r.get(key)
            if value_str is not None:
                return pickle.loads(value_str)
            value = func(*args, **kwds)
            # Randomly remove one entry if maxsize is reached
            if r.dbsize() >= maxsize:
                r.delete(r.randomkey())
            r.set(key, pickle.dumps(value))
            return value

        return wrapper

    return decorating_function
