

"""
InvalidService defines a invalid model handler for testing purpose.
"""


def handle(data, context):
    # This model is created to test reporting of an error in a batch of requests
    if data:
        context.set_response_status(code=507, idx=0)
    return ["Invalid response"]
