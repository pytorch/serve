
"""
InvalidService defines a invalid model handler for testing purpose.
"""

def handle(data, context):
    raise RuntimeError("Initialize failure.")
