

# pylint: disable=missing-docstring
import json


class Model(object):
    """
    Model is a part of the manifest.json. It defines the properties of the model such as name, version as weill
    as the entry point into the service code through the handler property
    """

    def __init__(self, model_name, serialized_file, handler, model_file=None, description=None, model_version=None,
                 extensions=None, source_vocab=None):
        self.model_name = model_name
        self.serialized_file = serialized_file.split("/")[-1]
        self.model_file = model_file
        self.description = description
        self.model_version = model_version
        self.extensions = extensions
        self.handler = handler.split("/")[-1]
        self.source_vocab = source_vocab
        self.model_dict = self.__to_dict__()

    def __to_dict__(self):
        model_dict = dict()

        model_dict['modelName'] = self.model_name

        model_dict['serializedFile'] = self.serialized_file

        model_dict['handler'] = self.handler

        if self.source_vocab:
            model_dict['sourceVocab'] = self.source_vocab.split("/")[-1]

        if self.model_file:
            model_dict['modelFile'] = self.model_file.split("/")[-1]

        if self.description:
            model_dict['description'] = self.description

        if self.model_version:
            model_dict['modelVersion'] = self.model_version

        if self.extensions:
            model_dict['extensions'] = self.extensions

        return model_dict

    def __str__(self):
        return json.dumps(self.model_dict)

    def __repr__(self):
        return json.dumps(self.model_dict)
