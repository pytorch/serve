

from collections import namedtuple

import pytest

from model_archiver.manifest_components.engine import EngineType
from model_archiver.manifest_components.manifest import RuntimeType
from model_archiver.model_packaging import generate_model_archive, package_model
from model_archiver.model_packaging_utils import ModelExportUtils


# noinspection PyClassHasNoInit
class TestModelPackaging:

    class Namespace:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

        def update(self, **kwargs):
            self.__dict__.update(kwargs)

    author = 'ABC'
    email = 'ABC@XYZ.com'
    engine = EngineType.MXNET.value
    model_name = 'my-model'
    model_file = 'my-model/'
    serialized_file = 'my-model/'
    handler = 'a.py::my-awesome-func'
    export_path = '/Users/dummyUser/'
    version = '1.0'
    source_vocab = None

    args = Namespace(author=author, email=email, engine=engine, model_name=model_name, handler=handler,
                     runtime=RuntimeType.PYTHON.value, model_file=model_file, serialized_file=serialized_file,
                     extra_files=None, export_path=export_path, force=False, archive_format="default", convert=False,
                     version=version, source_vocab=source_vocab)

    @pytest.fixture()
    def patches(self, mocker):
        Patches = namedtuple('Patches', ['arg_parse', 'export_utils', 'export_method'])
        patches = Patches(mocker.patch('model_archiver.model_packaging.ArgParser'),
                          mocker.patch('model_archiver.model_packaging.ModelExportUtils'),
                          mocker.patch('model_archiver.model_packaging.package_model'))

        return patches

    def test_gen_model_archive(self, patches):
        patches.arg_parse.export_model_args_parser.parse_args.return_value = self.args
        generate_model_archive()
        patches.export_method.assert_called()

    def test_export_model_method(self, patches):
        patches.export_utils.check_mar_already_exists.return_value = '/Users/dummyUser/'
        patches.export_utils.check_custom_model_types.return_value = '/Users/dummyUser', ['a.txt', 'b.txt']
        patches.export_utils.zip.return_value = None

        package_model(self.args, ModelExportUtils.generate_manifest_json(self.args))
        patches.export_utils.validate_inputs.assert_called()
        patches.export_utils.archive.assert_called()
        patches.export_utils.clean_temp_files.assert_called()

    def test_export_model_method_tar(self, patches):
        self.args.update(archive_format="tar")
        patches.export_utils.check_mar_already_exists.return_value = '/Users/dummyUser/'
        patches.export_utils.check_custom_model_types.return_value = '/Users/dummyUser', ['a.txt', 'b.txt']
        patches.export_utils.zip.return_value = None

        package_model(self.args, ModelExportUtils.generate_manifest_json(self.args))
        patches.export_utils.validate_inputs.assert_called()
        patches.export_utils.archive.assert_called()
        patches.export_utils.clean_temp_files.assert_called()

    def test_export_model_method_noarchive(self, patches):
        self.args.update(archive_format="no-archive")
        patches.export_utils.check_mar_already_exists.return_value = '/Users/dummyUser/'
        patches.export_utils.check_custom_model_types.return_value = '/Users/dummyUser', ['a.txt', 'b.txt']
        patches.export_utils.zip.return_value = None

        package_model(self.args, ModelExportUtils.generate_manifest_json(self.args))
        patches.export_utils.validate_inputs.assert_called()
        patches.export_utils.archive.assert_called()
        patches.export_utils.clean_temp_files.assert_called()
