# pylint: disable=missing-docstring
import json
import os


class Model(object):
    """
    Model is a part of the manifest.json. It defines the properties of the model such as name, version as weill
    as the entry point into the service code through the handler property
    """

    def __init__(
        self,
        model_name,
        serialized_file,
        handler,
        model_file=None,
        model_version=None,
        extensions=None,
        requirements_file=None,
        config_file=None,
    ):
        self.model_name = model_name
        self.serialized_file = None
        if serialized_file:
            self.serialized_file = os.path.basename(serialized_file)
        self.model_file = model_file
        self.model_version = model_version
        self.extensions = extensions
        self.handler = os.path.basename(handler)
        self.requirements_file = requirements_file
        self.config_file = None
        if config_file:
            self.config_file = os.path.basename(config_file)
        self.model_dict = self.__to_dict__()

    def __to_dict__(self):
        model_dict = {}

        model_dict["modelName"] = self.model_name

        if self.serialized_file:
            model_dict["serializedFile"] = self.serialized_file

        model_dict["handler"] = self.handler

        if self.model_file:
            model_dict["modelFile"] = self.model_file.split("/")[-1]

        if self.model_version:
            model_dict["modelVersion"] = self.model_version

        if self.extensions:
            model_dict["extensions"] = self.extensions

        if self.requirements_file:
            model_dict["requirementsFile"] = self.requirements_file.split("/")[-1]

        if self.config_file:
            model_dict["configFile"] = self.config_file

        return model_dict

    def __str__(self):
        return json.dumps(self.model_dict)

    def __repr__(self):
        return json.dumps(self.model_dict)
