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
from typing import Optional

from ts.metrics.metric_cache_yaml_impl import MetricsCacheYamlImpl
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
    def load(
        self,
        model_name: str,
        model_dir: str,
        handler: Optional[str] = None,
        gpu_id: Optional[int] = None,
        batch_size: Optional[int] = None,
        envelope: Optional[str] = None,
        limit_max_image_pixels: Optional[bool] = True,
    ):
        """
        Load model from file.

        :param model_name:
        :param model_dir:
        :param handler:
        :param gpu_id:
        :param batch_size:
        :param envelope:
        :param limit_max_image_pixels:
        :return: Model
        """
        # pylint: disable=unnecessary-pass
        pass


class TsModelLoader(ModelLoader):
    """
    TorchServe 1.0 Model Loader
    """

    def load(
        self,
        model_name: str,
        model_dir: str,
        handler: Optional[str] = None,
        gpu_id: Optional[int] = None,
        batch_size: Optional[int] = None,
        envelope: Optional[str] = None,
        limit_max_image_pixels: Optional[bool] = True,
        metrics_cache: Optional[MetricsCacheYamlImpl] = None,
    ) -> Service:
        """
        Load TorchServe 1.0 model from file.

        :param model_name:
        :param model_dir:
        :param handler:
        :param gpu_id:
        :param batch_size:
        :param envelope:
        :param limit_max_image_pixels:
        :param metrics_cache: MetricsCacheYamlImpl object
        :return:
        """
        logging.debug("Loading model - working dir: %s", os.getcwd())

        # Backwards Compatibility with releases <=0.6.0
        # Request ID is not set for model load requests
        # TODO: UUID serves as a temporary request ID for model load requests
        if metrics_cache is not None:
            metrics_cache.set_request_ids(str(uuid.uuid4()))

        manifest_file = os.path.join(model_dir, "MAR-INF", "MANIFEST.json")
        manifest = None
        if os.path.exists(manifest_file):
            with open(manifest_file) as f:
                manifest = json.load(f)

        function_name = None
        try:
            module, function_name = self._load_handler_file(handler)
        except ImportError:
            module = self._load_default_handler(handler)

        if module is None:
            raise ValueError(
                "Unable to load module {}, make sure it is added to python path".format(
                    handler
                )
            )
        function_name = function_name or "handle"

        if hasattr(module, function_name):
            entry_point, initialize_fn = self._get_function_entry_point(
                module, function_name
            )
        else:
            entry_point, initialize_fn = self._get_class_entry_point(module)

        if envelope is not None:
            envelope_class = self._load_default_envelope(envelope)
            if envelope_class is not None:
                envelope_instance = envelope_class(entry_point)
                entry_point = envelope_instance.handle

        service = Service(
            model_name,
            model_dir,
            manifest,
            entry_point,
            gpu_id,
            batch_size,
            limit_max_image_pixels,
            metrics_cache,
        )
        initialize_fn(service.context)

        return service

    def _load_handler_file(self, handler):
        temp = handler.split(":", 1)
        module_name = temp[0]
        if module_name.endswith(".py"):
            module_name = module_name[:-3]
        module_name = module_name.split("/")[-1]
        module = importlib.import_module(module_name)
        function_name = None if len(temp) == 1 else temp[1]
        return module, function_name

    def _load_default_handler(self, handler):
        module_name = ".{0}".format(handler)
        module = importlib.import_module(module_name, "ts.torch_handler")
        return module

    def _load_default_envelope(self, envelope):
        module_name = ".{0}".format(envelope)
        module = importlib.import_module(
            module_name, "ts.torch_handler.request_envelope"
        )
        envelope_class = list_classes_from_module(module)[0]
        return envelope_class

    def _get_function_entry_point(self, module, function_name):
        entry_point = getattr(module, function_name)
        initialize_fn = lambda ctx: entry_point(None, ctx)
        return entry_point, initialize_fn

    def _get_class_entry_point(self, module):
        model_class_definitions = list_classes_from_module(module)
        if len(model_class_definitions) != 1:
            raise ValueError(
                "Expected only one class in custom service code or a function entry point {}".format(
                    model_class_definitions
                )
            )

        model_class = model_class_definitions[0]
        model_service = model_class()

        if not hasattr(model_service, "handle"):
            raise ValueError(
                "Expect handle method in class {}".format(str(model_class))
            )

        return model_service.handle, model_service.initialize
