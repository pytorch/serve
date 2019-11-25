

"""
Dummy custom service which is class based
"""


# noinspection PyUnusedLocal
class CustomService(object):

    def initialize(self, context):
        pass

    # noinspection PyMethodMayBeStatic
    def handle(self, data, context):
        from ts.context import Context
        return ["OK"]
