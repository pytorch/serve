# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#     http://www.apache.org/licenses/LICENSE-2.0
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.
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
