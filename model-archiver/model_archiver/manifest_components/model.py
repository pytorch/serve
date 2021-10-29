# pylint: disable=missing-docstring
import json
import sys


class Model(object):
    """
    Model is a part of the manifest.json. It defines the properties of the model such as name, version as weill
    as the entry point into the service code through the handler property
    """

    def __init__(self, model_name, serialized_file, handler, model_file=None, model_version=None,
                 extensions=None, requirements_file=None):

        self.model_name = model_name
        self.serialized_file = None
        if serialized_file:
            if sys.platform.startswith('win32') and serialized_file.find("\\") != -1:
                self.serialized_file = serialized_file.split("\\")[-1]
            else:
                self.serialized_file = serialized_file.split("/")[-1]
        self.model_file = model_file
        self.model_version = model_version
        self.extensions = extensions
        if sys.platform.startswith('win32') and handler.find("\\") != -1:
            self.handler = handler.split("\\")[-1]
        else:
            self.handler = handler.split("/")[-1]
        self.requirements_file = requirements_file

        self.model_dict = self.__to_dict__()

    def __to_dict__(self):
        model_dict = dict()

        model_dict['modelName'] = self.model_name

        if self.serialized_file:
            model_dict['serializedFile'] = self.serialized_file

        model_dict['handler'] = self.handler

        if self.model_file:
            model_dict['modelFile'] = self.model_file.split("/")[-1]

        if self.model_version:
            model_dict['modelVersion'] = self.model_version

        if self.extensions:
            model_dict['extensions'] = self.extensions

        if self.requirements_file:
            model_dict['requirementsFile'] = self.requirements_file.split("/")[-1]

        return model_dict

    def __str__(self):
        return json.dumps(self.model_dict)

    def __repr__(self):
        return json.dumps(self.model_dict)
