

"""
Dummy custom service which is function based
"""

from mms.context import Context


# noinspection PyUnusedLocal
def infer(data, context):
    return isinstance(context, Context)
