

# pylint: disable=redefined-builtin
# pylint: disable=missing-docstring
import json
from enum import Enum


class RuntimeType(Enum):

    PYTHON = "python"
    PYTHON2 = "python2"
    PYTHON3 = "python3"


class Manifest(object):
    """
    The main manifest object which gets written into the model archive as MANIFEST.json
    """

    def __init__(self, runtime, model, engine=None, specification_version='1.0', implementation_version='1.0',
                 description=None, publisher=None, model_server_version='1.0', license=None, user_data=None):

        self.runtime = RuntimeType(runtime)
        self.engine = engine
        self.model = model
        self.publisher = publisher
        self.specification_version = specification_version
        self.implementation_version = implementation_version
        self.model_server_version = model_server_version
        self.license = license
        self.description = description
        self.user_data = user_data
        self.manifest_dict = self.__to_dict__()

    def __to_dict__(self):
        manifest_dict = dict()

        manifest_dict['runtime'] = self.runtime.value

        manifest_dict['model'] = self.model.__to_dict__()

        if self.engine is not None:
            manifest_dict['engine'] = self.engine.__to_dict__()

        if self.license is not None:
            manifest_dict['license'] = self.license

        if self.model_server_version is not None:
            manifest_dict['modelServerVersion'] = self.model_server_version

        if self.description is not None:
            manifest_dict['description'] = self.description

        if self.implementation_version is not None:
            manifest_dict['implementationVersion'] = self.implementation_version

        if self.specification_version is not None:
            manifest_dict['specificationVersion'] = self.specification_version

        if self.user_data is not None:
            manifest_dict['userData'] = self.user_data

        if self.publisher is not None:
            manifest_dict['publisher'] = self.publisher.__to_dict__()

        return manifest_dict

    def __str__(self):
        return json.dumps(self.manifest_dict, indent=2)

    def __repr__(self):
        return json.dumps(self.manifest_dict, indent=2)
