

"""
Dummy custom service which is class based
"""
from typing import List


# noinspection PyUnusedLocal
class CustomService(object):

    def initialize(self, context) -> None:
        pass

    # noinspection PyMethodMayBeStatic
    def handle(self, data, context) -> List[str]:
        from ts.context import Context
        return ["OK"]
