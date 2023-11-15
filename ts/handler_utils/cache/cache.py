import hashlib
import logging
import pickle
from functools import wraps

from ts.context import Context

logger = logging.getLogger(__name__)


class Cache:
    def __init__(self):
        self.client = None

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwds):
            # Removing Context objects from key hashing
            key = self._make_key(
                args=[arg for arg in args if not isinstance(arg, Context)],
                kwds={k: v for (k, v) in kwds.items() if not isinstance(v, Context)},
            )

            # if client is not set, execute handler as is
            if not self.client:
                return func(*args, **kwds)

            # Check if key in client
            value_str = self.client.get(key)
            if value_str is not None:
                logger.info("Cache hit")
                return pickle.loads(value_str)
            value = func(*args, **kwds)
            self.client.set(key, pickle.dumps(value))
            return value

        return wrapper

    def _no_op_decorator(self, func):
        @wraps(func)
        def wrapper(*args, **kwds):
            return func(*args, **kwds)

    def _make_key(self, args, kwds):
        key = args
        if kwds:
            key += (object(),)
            for item in kwds.items():
                key += item

        return hashlib.sha256(pickle.dumps(key)).hexdigest()
