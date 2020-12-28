

import json
import platform

import pytest
from collections import namedtuple
from workflow_archiver.workflow_packaging_utils import WorkflowExportUtils
from workflow_archiver.workflow_archiver_error import WorkflowArchiverError


# noinspection PyClassHasNoInit
def _validate_war(patches):
    if platform.system() == "Windows":
        patches.path_exists.assert_called_once_with("/Users/dummyUser\\some-workflow.war")
    else:
        patches.path_exists.assert_called_once_with('/Users/dummyUser/some-workflow.war')


# noinspection PyClassHasNoInit
class TestExportWorkflowUtils:

    # noinspection PyClassHasNoInit
    class TestWarExistence:

        @pytest.fixture()
        def patches(self, mocker):
            Patches = namedtuple('Patches', ['getcwd', 'path_exists'])
            patches = Patches(mocker.patch('os.getcwd'), mocker.patch('os.path.exists'))
            patches.getcwd.return_value = '/Users/dummyUser'
            return patches

        def test_export_file_is_none(self, patches):
            patches.path_exists.return_value = False
            ret_val = WorkflowExportUtils.check_war_already_exists('some-workflow', None, False)
            _validate_war(patches)
            assert ret_val == "/Users/dummyUser"

        def test_export_file_is_not_none(self, patches):
            patches.path_exists.return_value = False
            WorkflowExportUtils.check_war_already_exists('some-workflow', '/Users/dummyUser/', False)
            patches.path_exists.assert_called_once_with('/Users/dummyUser/some-workflow.war')

        def test_export_file_already_exists_with_override(self, patches):
            patches.path_exists.return_value = True
            WorkflowExportUtils.check_war_already_exists('some-workflow', None, True)
            _validate_war(patches)

        def test_export_file_already_exists_with_override_false(self, patches):
            patches.path_exists.return_value = True
            with pytest.raises(WorkflowArchiverError):
                WorkflowExportUtils.check_war_already_exists('some-workflow', None, False)
            _validate_war(patches)

    # noinspection PyClassHasNoInit
    class TestCustomWorkflowTypes:

        workflow_path = '/Users/dummyUser'

        @pytest.fixture()
        def patches(self, mocker):
            Patches = namedtuple('Patches', ['utils', 'listdir'])
            patch = Patches(mocker.patch('workflow_archiver.workflow_packaging_utils.WorkflowExportUtils'),
                            mocker.patch('os.listdir'))

            patch.listdir.return_value = {'a', 'b', 'c'}
            return patch

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
            WorkflowExportUtils.clean_temp_files(temp_files)

            patches.remove.assert_called()
            assert patches.remove.call_count == len(temp_files)

    # noinspection PyClassHasNoInit
    class TestGenerateManifestProps:

        class Namespace:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        workflow_name = 'my-workflow'
        handler = 'a.py'
        spec_file = 'spec.yaml'
        version = "1.0"

        args = Namespace(workflow_name=workflow_name, handler=handler, spec_file=spec_file)

        def test_workflow(self):
            workflow = WorkflowExportUtils.generate_workflow(self.args)
            assert workflow.workflow_name == self.workflow_name
            assert workflow.handler == self.handler

        def test_manifest_json(self):
            manifest = WorkflowExportUtils.generate_manifest_json(self.args)
            manifest_json = json.loads(manifest)
            assert 'workflow' in manifest_json
            assert 'license' not in manifest_json

    # noinspection PyClassHasNoInit
    class TestWorkflowNameRegEx:

        def test_regex_pass(self):
            workflow_names = ['my-awesome-workflow', 'Aa.workflow', 'a', 'aA.workflow', 'a1234.workflow',
                              'a-A-A.workflow', '123-abc']
            for m in workflow_names:
                WorkflowExportUtils.check_workflow_name_regex_or_exit(m)

        def test_regex_fail(self):
            workflow_names = ['abc%', '123$abc', 'abc!123', '@123', '(workflow', 'wrokflow)',
                           '12*workflow-a.workflow', '##.workflow', '-.workflow']
            for m in workflow_names:
                with pytest.raises(WorkflowArchiverError):
                    WorkflowExportUtils.check_workflow_name_regex_or_exit(m)

    # noinspection PyClassHasNoInit
    class TestFileFilter:

        files_to_exclude = {'abc.onnx'}

        def test_with_return_false(self):
            assert WorkflowExportUtils.file_filter('abc.onnx', self.files_to_exclude) is False

        def test_with_pyc(self):
            assert WorkflowExportUtils.file_filter('abc.pyc', self.files_to_exclude) is False

        def test_with_ds_store(self):
            assert WorkflowExportUtils.file_filter('.DS_Store', self.files_to_exclude) is False

        def test_with_return_true(self):
            assert WorkflowExportUtils.file_filter('abc.mxnet', self.files_to_exclude) is True

    # noinspection PyClassHasNoInit
    class TestDirectoryFilter:

        unwanted_dirs = {'__MACOSX', '__pycache__'}

        def test_with_unwanted_dirs(self):
            assert WorkflowExportUtils.directory_filter('__MACOSX', self.unwanted_dirs) is False

        def test_with_starts_with_dot(self):
            assert WorkflowExportUtils.directory_filter('.gitignore', self.unwanted_dirs) is False

        def test_with_return_true(self):
            assert WorkflowExportUtils.directory_filter('my-workflow', self.unwanted_dirs) is True
