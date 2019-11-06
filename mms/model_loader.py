# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#     http://www.apache.org/licenses/LICENSE-2.0
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

"""
Model loader.
"""
import importlib
import inspect
import json
import logging
import os
import sys
import uuid
from abc import ABCMeta, abstractmethod

from builtins import str

from mms.metrics.metrics_store import MetricsStore
from mms.service import Service


class ModelLoaderFactory(object):
    """
    ModelLoaderFactory
    """

    @staticmethod
    def get_model_loader(model_dir):
        manifest_file = os.path.join(model_dir, "MAR-INF/MANIFEST.json")
        if os.path.exists(manifest_file):
            return MmsModelLoader()
        elif os.path.exists(os.path.join(model_dir, "MANIFEST.json")):
            return LegacyModelLoader()
        else:
            return MmsModelLoader()


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

    @staticmethod
    def list_model_services(module, parent_class=None):
        """
        Parse user defined module to get all model service classes in it.

        :param module:
        :param parent_class:
        :return: List of model service class definitions
        """

        # Parsing the module to get all defined classes
        classes = [cls[1] for cls in inspect.getmembers(module, lambda member: inspect.isclass(member) and
                                                        member.__module__ == module.__name__)]
        # filter classes that is subclass of parent_class
        if parent_class is not None:
            return [c for c in classes if issubclass(c, parent_class)]

        return classes


class MmsModelLoader(ModelLoader):
    """
    MMS 1.0 Model Loader
    """

    def load(self, model_name, model_dir, handler, gpu_id, batch_size):
        """
        Load MMS 1.0 model from file.

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

        temp = handler.split(":", 1)
        module_name = temp[0]
        function_name = None if len(temp) == 1 else temp[1]
        if module_name.endswith(".py"):
            module_name = module_name[:-3]
        module_name = module_name.split("/")[-1]
        module = importlib.import_module(module_name)
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
            model_class_definitions = ModelLoader.list_model_services(module)
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


class LegacyModelLoader(ModelLoader):
    """
    MMS 0.4 Model Loader
    """

    def load(self, model_name, model_dir, handler, gpu_id, batch_size):
        """
        Load MMS 0.3 model from file.

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

        from mms.model_service.mxnet_model_service import SingleNodeService

        model_class_definitions = ModelLoader.list_model_services(module, SingleNodeService)
        module_class = model_class_definitions[0]

        module = module_class(model_name, model_dir, manifest, gpu_id)
        service = Service(model_name, model_dir, manifest, module.handle, gpu_id, batch_size)

        module.initialize(service.context)

        return service
