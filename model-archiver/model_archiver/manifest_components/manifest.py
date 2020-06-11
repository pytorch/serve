# pylint: disable=redefined-builtin
# pylint: disable=missing-docstring
from datetime import datetime
import json
from model_archiver import __version__
from enum import Enum


class RuntimeType(Enum):

    PYTHON = "python"
    PYTHON2 = "python2"
    PYTHON3 = "python3"


class Manifest(object):
    """
    The main manifest object which gets written into the model archive as MANIFEST.json
    """

    def __init__(self, runtime, model, description=None,license=None, user_data=None):

        self.creation_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        self.runtime = RuntimeType(runtime)
        self.model = model
        self.archiver_version = __version__
        self.license = license
        self.description = description
        self.user_data = user_data
        self.manifest_dict = self.__to_dict__()

    def __to_dict__(self):
        manifest_dict = dict()

        manifest_dict['createdOn'] = self.creation_time

        manifest_dict['runtime'] = self.runtime.value

        manifest_dict['model'] = self.model.__to_dict__()

        if self.license is not None:
            manifest_dict['license'] = self.license

        if self.archiver_version is not None:
            manifest_dict['archiverVersion'] = self.archiver_version

        if self.description is not None:
            manifest_dict['description'] = self.description

        if self.user_data is not None:
            manifest_dict['userData'] = self.user_data

        return manifest_dict

    def __str__(self):
        return json.dumps(self.manifest_dict, indent=2)

    def __repr__(self):
        return json.dumps(self.manifest_dict, indent=2)
