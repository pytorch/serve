"""
Base class for all RequestEnvelope.

A request envelope reformats the inputs/outputs of a call to a handler.
It translates from formats specific to a model orchestrator like Seldon or
KServe to a set of flat Python items, and vice versa.
"""

from abc import ABC, abstractmethod


class BaseEnvelope(ABC):
    """
    Interface for all envelopes.
    Derive from this class, replacing the abstract methods
    """
    def __init__(self, handle_fn):
        self._handle_fn = handle_fn
        self.context = None
    def handle(self, data, context):
        """
        The Input Requests and Response are handled here.
        """
        self.context = context
        if data:
            data = self.parse_input(data)

        results = self._handle_fn(data, context)

        if results:
            results = self.format_output(results)

        return results

    @abstractmethod
    def parse_input(self, data):
        pass

    @abstractmethod
    def format_output(self, data):
        pass
