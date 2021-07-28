

from collections import namedtuple

import pytest

from workflow_archiver.workflow_packaging import generate_workflow_archive, package_workflow
from workflow_archiver.workflow_packaging_utils import WorkflowExportUtils


# noinspection PyClassHasNoInit
class TestWorkflowPackaging:

    class Namespace:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

        def update(self, **kwargs):
            self.__dict__.update(kwargs)

    workflow_name = 'my-workflow'
    spec_file = 'my-workflow/'
    handler = 'a.py'
    export_path = '/Users/dummyUser/'

    args = Namespace(workflow_name=workflow_name, handler=handler, spec_file=spec_file, export_path=export_path,
                     force=False, convert=False, extra_files=None)

    @pytest.fixture()
    def patches(self, mocker):
        Patches = namedtuple('Patches', ['arg_parse', 'export_utils', 'export_method'])
        patches = Patches(mocker.patch('workflow_archiver.workflow_packaging.ArgParser'),
                          mocker.patch('workflow_archiver.workflow_packaging.WorkflowExportUtils'),
                          mocker.patch('workflow_archiver.workflow_packaging.package_workflow'))

        return patches

    def test_gen_workflow_archive(self, patches):
        patches.arg_parse.workflow_archiver_args_parser.parse_args.return_value = self.args
        generate_workflow_archive()
        patches.export_method.assert_called()

    def test_export_workflow_method(self, patches):
        patches.export_utils.check_war_already_exists.return_value = '/Users/dummyUser/'
        patches.export_utils.zip.return_value = None

        package_workflow(self.args, WorkflowExportUtils.generate_manifest_json(self.args))
        patches.export_utils.validate_inputs.assert_called()
        patches.export_utils.archive.assert_called()
        patches.export_utils.clean_temp_files.assert_called()
