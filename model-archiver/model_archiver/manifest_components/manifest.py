# pylint: disable=redefined-builtin
# pylint: disable=missing-docstring
import json
from datetime import datetime
from enum import Enum

from model_archiver import __version__


class RuntimeType(Enum):
    PYTHON = "python"
    PYTHON3 = "python3"


class Manifest(object):
    """
    The main manifest object which gets written into the model archive as MANIFEST.json
    """

    def __init__(self, runtime, model):
        self.creation_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        self.runtime = RuntimeType(runtime)
        self.model = model
        self.archiver_version = __version__
        self.manifest_dict = self.__to_dict__()

    def __to_dict__(self):
        manifest_dict = {}

        manifest_dict["createdOn"] = self.creation_time

        manifest_dict["runtime"] = self.runtime.value

        manifest_dict["model"] = self.model.__to_dict__()

        if self.archiver_version is not None:
            manifest_dict["archiverVersion"] = self.archiver_version

        return manifest_dict

    def __str__(self):
        return json.dumps(self.manifest_dict, indent=2)

    def __repr__(self):
        return json.dumps(self.manifest_dict, indent=2)
