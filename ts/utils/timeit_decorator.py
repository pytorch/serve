
"""
timeit decorator
"""

import time
from functools import wraps


def timeit(func):
    """
    Use this decorator on a method to find it's execution time.
    :param func:
    :return:
    """
    @wraps(func)
    def time_and_log(*args, **kwargs):
        start = time.time()
        start_cpu = time.clock()
        result = func(*args, **kwargs)
        end = time.time()
        end_cpu = time.clock()
        print("func: %r took a total of %2.4f sec to run and %2.4f sec of CPU time\n",
              (func.__name__, (end-start), (end_cpu - start_cpu)))
        return result
    return time_and_log
