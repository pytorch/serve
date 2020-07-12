"""
Default class for all handlers. Grabs the data out of the request body.
"""

from .base import BaseEnvelope

class BodyEnvelope(BaseEnvelope):
    """
    Gets the key "body" from the input data, returns a raw list
    """
    def parse_input(self, data):
        return [row.get("data") or row.get("body") for row in data]

    def format_output(self, data):
        return data
