

import json
import os
import pytest
from collections import namedtuple
from model_archiver.model_packaging_utils import ModelExportUtils
from model_archiver.manifest_components.engine import EngineType
from model_archiver.manifest_components.manifest import RuntimeType
from model_archiver.model_archiver_error import ModelArchiverError


# noinspection PyClassHasNoInit
class TestExportModelUtils:

    # noinspection PyClassHasNoInit
    class TestMarExistence:

        @pytest.fixture()
        def patches(self, mocker):
            Patches = namedtuple('Patches', ['getcwd', 'path_exists'])
            patches = Patches(mocker.patch('os.getcwd'), mocker.patch('os.path.exists'))

            patches.getcwd.return_value = '/Users/dummyUser'

            return patches

        def test_export_file_is_none(self, patches):
            patches.path_exists.return_value = False
            ret_val = ModelExportUtils.check_mar_already_exists('some-model', None, False)

            patches.path_exists.assert_called_once_with("/Users/dummyUser/some-model.mar")
            assert ret_val == "/Users/dummyUser"

        def test_export_file_is_not_none(self, patches):
            patches.path_exists.return_value = False
            ModelExportUtils.check_mar_already_exists('some-model', '/Users/dummyUser/', False)

            patches.path_exists.assert_called_once_with('/Users/dummyUser/some-model.mar')

        def test_export_file_already_exists_with_override(self, patches):
            patches.path_exists.return_value = True

            ModelExportUtils.check_mar_already_exists('some-model', None, True)

            patches.path_exists.assert_called_once_with('/Users/dummyUser/some-model.mar')

        def test_export_file_already_exists_with_override_false(self, patches):
            patches.path_exists.return_value = True

            with pytest.raises(ModelArchiverError):
                ModelExportUtils.check_mar_already_exists('some-model', None, False)

            patches.path_exists.assert_called_once_with('/Users/dummyUser/some-model.mar')

        def test_export_file_is_none_tar(self, patches):
            patches.path_exists.return_value = False
            ret_val = ModelExportUtils.check_mar_already_exists('some-model', None, False, archive_format='tgz')

            patches.path_exists.assert_called_once_with("/Users/dummyUser/some-model.tar.gz")
            assert ret_val == "/Users/dummyUser"

        def test_export_file_is_none_tar(self, patches):
            patches.path_exists.return_value = False
            ret_val = ModelExportUtils.check_mar_already_exists('some-model', None, False, archive_format='no-archive')

            patches.path_exists.assert_called_once_with("/Users/dummyUser/some-model")
            assert ret_val == "/Users/dummyUser"

    # noinspection PyClassHasNoInit
    class TestArchiveTypes:
        def test_archive_types(self):
            from model_archiver.model_packaging_utils import archiving_options as ar_opts
            assert ar_opts.get("tgz") == ".tar.gz"
            assert ar_opts.get("no-archive") == ""
            assert ar_opts.get("default") == ".mar"
            assert len(ar_opts) == 3

    # noinspection PyClassHasNoInit
    class TestCustomModelTypes:

        model_path = '/Users/dummyUser'

        @pytest.fixture()
        def patches(self, mocker):
            Patches = namedtuple('Patches', ['utils', 'listdir'])
            patch = Patches(mocker.patch('model_archiver.model_packaging_utils.ModelExportUtils'),
                            mocker.patch('os.listdir'))

            patch.listdir.return_value = {'a', 'b', 'c'}
            return patch

    # noinspection PyClassHasNoInit
    class TestFindUnique:

        def test_with_count_zero(self):
            files = ['a.txt', 'b.txt', 'c.txt']
            suffix = '.mxnet'
            val = ModelExportUtils.find_unique(files, suffix)
            assert val is None

        def test_with_count_one(self):
            files = ['a.mxnet', 'b.txt', 'c.txt']
            suffix = '.mxnet'
            val = ModelExportUtils.find_unique(files, suffix)
            assert val == 'a.mxnet'

        def test_with_exit(self):
            files = ['a.onnx', 'b.onnx', 'c.txt']
            suffix = '.onnx'
            with pytest.raises(ModelArchiverError):
                ModelExportUtils.find_unique(files, suffix)

    # noinspection PyClassHasNoInit
    class TestCleanTempFiles:

        @pytest.fixture()
        def patches(self, mocker):
            Patches = namedtuple('Patches', ['remove'])
            patches = Patches(mocker.patch('os.remove'))

            patches.remove.return_value = True
            return patches

        def test_clean_call(self, patches):
            temp_files = ['a', 'b', 'c']
            ModelExportUtils.clean_temp_files(temp_files)

            patches.remove.assert_called()
            assert patches.remove.call_count == len(temp_files)

    # noinspection PyClassHasNoInit
    class TestGenerateManifestProps:

        class Namespace:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        author = 'ABC'
        email = 'ABC@XYZ.com'
        engine = EngineType.MXNET.value
        model_name = 'my-model'
        handler = 'a.py::my-awesome-func'
        serialized_file = 'model.pt'
        model_file = 'model.pt'
        version = "1.0"
        source_vocab = None

        args = Namespace(author=author, email=email, engine=engine, model_name=model_name, handler=handler,
                         runtime=RuntimeType.PYTHON.value, serialized_file=serialized_file, model_file=model_file,
                         version=version, source_vocab=source_vocab)

        def test_publisher(self):
            pub = ModelExportUtils.generate_publisher(self.args)
            assert pub.email == self.email
            assert pub.author == self.author

        def test_engine(self):
            eng = ModelExportUtils.generate_engine(self.args)
            assert eng.engine_name == EngineType(self.engine)

        def test_model(self):
            mod = ModelExportUtils.generate_model(self.args)
            assert mod.model_name == self.model_name
            assert mod.handler == self.handler

        def test_manifest_json(self):
            manifest = ModelExportUtils.generate_manifest_json(self.args)
            manifest_json = json.loads(manifest)
            assert manifest_json['runtime'] == RuntimeType.PYTHON.value
            assert 'engine' in manifest_json
            assert 'model' in manifest_json
            assert 'publisher' in manifest_json
            assert 'license' not in manifest_json

    # noinspection PyClassHasNoInit
    class TestModelNameRegEx:

        def test_regex_pass(self):
            model_names = ['my-awesome-model', 'Aa.model', 'a', 'aA.model', 'a1234.model', 'a-A-A.model', '123-abc']
            for m in model_names:
                ModelExportUtils.check_model_name_regex_or_exit(m)

        def test_regex_fail(self):
            model_names = ['abc%', '123$abc', 'abc!123', '@123', '(model', 'mdoel)',
                           '12*model-a.model', '##.model', '-.model']
            for m in model_names:
                with pytest.raises(ModelArchiverError):
                    ModelExportUtils.check_model_name_regex_or_exit(m)

    # noinspection PyClassHasNoInit
    class TestFileFilter:

        files_to_exclude = {'abc.onnx'}

        def test_with_return_false(self):
            assert ModelExportUtils.file_filter('abc.onnx', self.files_to_exclude) is False

        def test_with_pyc(self):
            assert ModelExportUtils.file_filter('abc.pyc', self.files_to_exclude) is False

        def test_with_ds_store(self):
            assert ModelExportUtils.file_filter('.DS_Store', self.files_to_exclude) is False

        def test_with_return_true(self):
            assert ModelExportUtils.file_filter('abc.mxnet', self.files_to_exclude) is True

    # noinspection PyClassHasNoInit
    class TestDirectoryFilter:

        unwanted_dirs = {'__MACOSX', '__pycache__'}

        def test_with_unwanted_dirs(self):
            assert ModelExportUtils.directory_filter('__MACOSX', self.unwanted_dirs) is False

        def test_with_starts_with_dot(self):
            assert ModelExportUtils.directory_filter('.gitignore', self.unwanted_dirs) is False

        def test_with_return_true(self):
            assert ModelExportUtils.directory_filter('my-model', self.unwanted_dirs) is True
