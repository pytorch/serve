# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#     http://www.apache.org/licenses/LICENSE-2.0
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

from ts.model_service.model_service import SingleNodeService

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
