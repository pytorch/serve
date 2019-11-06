# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#     http://www.apache.org/licenses/LICENSE-2.0
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

# pylint: disable=missing-docstring
import json
from enum import Enum


class EngineType(Enum):
    MXNET = "MXNet"

    # TODO Add more engines here as and when MMS supports more DL Frameworks


class Engine(object):
    """
    Engine is a part of the final manifest.json. It defines which framework to run the inference on
    """

    def __init__(self, engine_name, engine_version=None):
        self.engine_name = EngineType(engine_name)
        self.engine_version = engine_version

        self.engine_dict = self.__to_dict__()

    def __to_dict__(self):
        engine_dict = dict()
        engine_dict['engineName'] = self.engine_name.value

        if self.engine_version is not None:
            engine_dict['engineVersion'] = self.engine_version

        return engine_dict

    def __str__(self):
        return json.dumps(self.engine_dict)

    def __repr__(self):
        return json.dumps(self.engine_dict)
