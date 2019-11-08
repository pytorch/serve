

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
