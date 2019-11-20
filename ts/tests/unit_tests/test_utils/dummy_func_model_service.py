

"""
Dummy custom service which is function based
"""

from ts.context import Context


# noinspection PyUnusedLocal
def infer(data, context):
    return isinstance(context, Context)
