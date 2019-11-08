

from mms.model_service.model_service import SingleNodeService

"""
This file is a dummy file for the purpose of unit-testing test_service_manager.py
"""


class DummyNodeService(SingleNodeService):
    def _inference(self, data):
        pass

    def signature(self):
        pass

    def ping(self):
        pass

    def inference(self):
        pass


class SomeOtherClass:
    def __init__(self):
        pass
