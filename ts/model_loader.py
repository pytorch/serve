

"""
Model loader.
"""
import importlib
import json
import logging
import os
import uuid
from abc import ABCMeta, abstractmethod

from builtins import str

from ts.metrics.metrics_store import MetricsStore
from ts.service import Service
from .utils.util import list_classes_from_module


class ModelLoaderFactory(object):
    """
    ModelLoaderFactory
    """

    @staticmethod
    def get_model_loader():
        return TsModelLoader()


class ModelLoader(object):
    """
    Base Model Loader class.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def load(self, model_name, model_dir, handler, gpu_id, batch_size):
        """
        Load model from file.

        :param model_name:
        :param model_dir:
        :param handler:
        :param gpu_id:
        :param batch_size:
        :return: Model
        """
        # pylint: disable=unnecessary-pass
        pass


class TsModelLoader(ModelLoader):
    """
    TorchServe 1.0 Model Loader
    """

    def load(self, model_name, model_dir, handler, gpu_id, batch_size):
        """
        Load TorchServe 1.0 model from file.

        :param model_name:
        :param model_dir:
        :param handler:
        :param gpu_id:
        :param batch_size:
        :return:
        """
        logging.debug("Loading model - working dir: %s", os.getcwd())
        # TODO: Request ID is not given. UUID is a temp UUID.
        metrics = MetricsStore(uuid.uuid4(), model_name)
        manifest_file = os.path.join(model_dir, "MAR-INF/MANIFEST.json")
        manifest = None
        if os.path.exists(manifest_file):
            with open(manifest_file) as f:
                manifest = json.load(f)

        try:
            temp = handler.split(":", 1)
            module_name = temp[0]
            function_name = None if len(temp) == 1 else temp[1]
            if module_name.endswith(".py"):
                module_name = module_name[:-3]
            module_name = module_name.split("/")[-1]
            module = importlib.import_module(module_name)
            # pylint: disable=unused-variable
        except ImportError as e:
            module_name = ".{0}".format(handler)
            module = importlib.import_module(module_name, 'ts.torch_handler')
            function_name = None

        if module is None:
            raise ValueError("Unable to load module {}, make sure it is added to python path".format(module_name))
        if function_name is None:
            function_name = "handle"
        if hasattr(module, function_name):
            entry_point = getattr(module, function_name)
            service = Service(model_name, model_dir, manifest, entry_point, gpu_id, batch_size)

            service.context.metrics = metrics
            # initialize model at load time
            entry_point(None, service.context)
        else:
            model_class_definitions = list_classes_from_module(module)
            if len(model_class_definitions) != 1:
                raise ValueError("Expected only one class in custom service code or a function entry point {}".format(
                    model_class_definitions))

            model_class = model_class_definitions[0]
            model_service = model_class()
            handle = getattr(model_service, "handle")
            if handle is None:
                raise ValueError("Expect handle method in class {}".format(str(model_class)))

            service = Service(model_name, model_dir, manifest, model_service.handle, gpu_id, batch_size)
            initialize = getattr(model_service, "initialize")
            if initialize is not None:
                model_service.initialize(service.context)
            else:
                raise ValueError("Expect initialize method in class {}".format(str(model_class)))

        return service
