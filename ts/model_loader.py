

"""
Model loader.
"""
import importlib
import json
import logging
import os
import sys
import uuid
from abc import ABCMeta, abstractmethod

from builtins import str

from ts.metrics.metrics_store import MetricsStore
from ts.service import Service


class ModelLoaderFactory(object):
    """
    ModelLoaderFactory
    """

    @staticmethod
    def get_model_loader(model_dir):
        manifest_file = os.path.join(model_dir, "MAR-INF/MANIFEST.json")
        if os.path.exists(manifest_file):
            return TsModelLoader()
        elif os.path.exists(os.path.join(model_dir, "MANIFEST.json")):
            return LegacyModelLoader()
        else:
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
    TS 1.0 Model Loader
    """

    def load(self, model_name, model_dir, handler, gpu_id, batch_size):
        """
        Load TS 1.0 model from file.

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

        if ':' in handler:
            temp = handler.split(":", 1)
            module_name = temp[0]
            function_name = None if len(temp) == 1 else temp[1]
            if module_name.endswith(".py"):
                module_name = module_name[:-3]
            module_name = module_name.split("/")[-1]
            module = importlib.import_module(module_name)
        else:
            from ts.torch_hanlder import image_classifier
            module = image_classifier
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
            from .utils import list_classes_from_module
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
                # noinspection PyBroadException
                try:
                    model_service.initialize(service.context)
                    # pylint: disable=broad-except
                except Exception:
                    # noinspection PyBroadException
                    try:
                        sys.exc_clear()
                        # pylint: disable=broad-except
                    except Exception:
                        pass

        return service

#TODO Shall we remove this?
class LegacyModelLoader(ModelLoader):
    """
    TS 0.4 Model Loader
    """

    def load(self, model_name, model_dir, handler, gpu_id, batch_size):
        """
        Load TS 0.3 model from file.

        :param model_name:
        :param model_dir:
        :param handler:
        :param gpu_id:
        :param batch_size:
        :return:
        """
        manifest_file = os.path.join(model_dir, "MANIFEST.json")

        manifest = None
        if os.path.isfile(manifest_file):
            with open(manifest_file) as f:
                manifest = json.load(f)
        if not handler.endswith(".py"):
            handler = handler + ".py"

        service_file = os.path.join(model_dir, handler)
        name = os.path.splitext(os.path.basename(service_file))[0]
        if sys.version_info[0] > 2:
            from importlib import util

            spec = util.spec_from_file_location(name, service_file)
            module = util.module_from_spec(spec)
            spec.loader.exec_module(module)
        else:
            import imp
            module = imp.load_source(name, service_file)

        if module is None:
            raise ValueError("Unable to load module {}".format(service_file))

        from ts.model_service.model_service import SingleNodeService

        model_class_definitions = ModelLoader.list_model_services(module, SingleNodeService)
        module_class = model_class_definitions[0]

        module = module_class(model_name, model_dir, manifest, gpu_id)
        service = Service(model_name, model_dir, manifest, module.handle, gpu_id, batch_size)

        module.initialize(service.context)

        return service
