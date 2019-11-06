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


class Model(object):
    """
    Model is a part of the manifest.json. It defines the properties of the model such as name, version as weill
    as the entry point into the service code through the handler property
    """

    def __init__(self, model_name, handler, description=None, model_version=None, extensions=None):
        self.model_name = model_name
        self.description = description
        self.model_version = model_version
        self.extensions = extensions
        self.handler = handler
        self.model_dict = self.__to_dict__()

    def __to_dict__(self):
        model_dict = dict()

        model_dict['modelName'] = self.model_name

        model_dict['handler'] = self.handler

        if self.description is not None:
            model_dict['description'] = self.description

        if self.model_version is not None:
            model_dict['modelVersion'] = self.model_version

        if self.extensions is not None:
            model_dict['extensions'] = self.extensions

        return model_dict

    def __str__(self):
        return json.dumps(self.model_dict)

    def __repr__(self):
        return json.dumps(self.model_dict)
