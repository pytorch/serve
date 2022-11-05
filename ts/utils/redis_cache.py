import logging
import pickle
from functools import wraps

import redis

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
    r = redis.Redis(host=host, port=port, db=db)
    try:
        r.ping()
    except ConnectionError as e:
        logging.info(
            f"Cannot connect to a redis server, ensure a server is running on {host}:{port}"
        )
        raise e

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
                return pickle.loads(value_str)  # might need to decode
            value = func(*args, **kwds)
            # Randomly remove one entry if maxsize is reached
            if r.dbsize() >= maxsize:
                r.delete(r.randomkey())
            r.set(key, pickle.dumps(value))
            return value

        return wrapper

    return decorating_function
