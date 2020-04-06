

import importlib
import inspect
import os
import sys
import types
from collections import namedtuple

import mock
import pytest

from ts.model_loader import TsModelLoader
from ts.model_loader import ModelLoaderFactory
from ts.model_service.model_service import SingleNodeService
from ts.utils.util import list_classes_from_module


# noinspection PyClassHasNoInit
# @pytest.mark.skip(reason="Disabling it currently until the PR #467 gets merged")
class TestModelFactory:
    def test_model_loader_factory(self):
        model_loader = ModelLoaderFactory.get_model_loader()

        assert isinstance(model_loader, TsModelLoader)


# noinspection PyClassHasNoInit
class TestListModels:

    def test_list_models_legacy(self):
        sys.path.append(os.path.abspath('ts/tests/unit_tests/model_service/dummy_model'))
        module = importlib.import_module('dummy_model_service')
        classes = list_classes_from_module(module, SingleNodeService)
        assert len(classes) == 1
        assert issubclass(classes[0], SingleNodeService)

    def test_list_models(self):
        sys.path.append(os.path.abspath('ts/tests/unit_tests/test_utils/'))
        module = importlib.import_module('dummy_class_model_service')
        classes = list_classes_from_module(module)
        assert len(classes) == 1
        assert classes[0].__name__ == 'CustomService'


# noinspection PyProtectedMember
# noinspection PyClassHasNoInit
class TestLoadModels:
    model_name = 'testmodel'
    model_dir = os.path.abspath('ts/tests/unit_tests/model_service/dummy_model')
    mock_manifest = '{"Model":{"Service":"dummy_class_model_service.py",' \
                    '"Signature":"signature.json","Model-Name":"testmodel"}}'

    @pytest.fixture()
    def patches(self, mocker):
        Patches = namedtuple('Patches', ['mock_open', 'os_path', "is_file", "open_signature"])
        patches = Patches(
            mocker.patch('ts.model_loader.open'),
            mocker.patch('os.path.exists'),
            mocker.patch('os.path.isfile'),
            mocker.patch('ts.model_service.model_service.open')
        )
        return patches

    def test_load_class_model(self, patches):
        patches.mock_open.side_effect = [mock.mock_open(read_data=self.mock_manifest).return_value]
        sys.path.append(os.path.abspath('ts/tests/unit_tests/test_utils/'))
        patches.os_path.return_value = True
        handler = 'dummy_class_model_service'
        model_loader = ModelLoaderFactory.get_model_loader()
        service = model_loader.load(self.model_name, self.model_dir, handler, 0, 1)

        assert inspect.ismethod(service._entry_point)

    def test_load_func_model(self, patches):
        patches.mock_open.side_effect = [mock.mock_open(read_data=self.mock_manifest).return_value]
        sys.path.append(os.path.abspath('ts/tests/unit_tests/test_utils/'))
        patches.os_path.return_value = True
        handler = 'dummy_func_model_service:infer'
        model_loader = ModelLoaderFactory.get_model_loader()
        service = model_loader.load(self.model_name, self.model_dir, handler, 0, 1)

        assert isinstance(service._entry_point, types.FunctionType)
        assert service._entry_point.__name__ == 'infer'

    def test_load_func_model_with_error(self, patches):
        patches.mock_open.side_effect = [mock.mock_open(read_data=self.mock_manifest).return_value]
        sys.path.append(os.path.abspath('ts/tests/unit_tests/test_utils/'))
        patches.os_path.return_value = True
        handler = 'dummy_func_model_service:wrong'
        model_loader = ModelLoaderFactory.get_model_loader()
        with pytest.raises(ValueError, match=r"Expected only one class .*"):
            model_loader.load(self.model_name, self.model_dir, handler, 0, 1)

    def test_load_model_with_error(self, patches):
        patches.mock_open.side_effect = [
            mock.mock_open(read_data='{"test" : "h"}').return_value]
        sys.path.append(os.path.abspath('ts/tests/unit_tests/test_utils/'))
        patches.os_path.return_value = True
        handler = 'dummy_func_model_service'
        model_loader = ModelLoaderFactory.get_model_loader()
        with pytest.raises(ValueError, match=r"Expected only one class .*"):
            model_loader.load(self.model_name, self.model_dir, handler, 0, 1)
