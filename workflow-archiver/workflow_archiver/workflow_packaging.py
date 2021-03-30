
"""
Command line interface to export workflow files to be used for inference by TorchServe
"""

import logging
import sys
from .arg_parser import ArgParser
from .workflow_packaging_utils import WorkflowExportUtils
from .workflow_archiver_error import WorkflowArchiverError


def package_workflow(args, manifest):
    """
    Internal helper for the exporting workflow command line interface.
    """
    workflow_spec_file = args.spec_file
    workflow_name = args.workflow_name
    handler = args.handler
    export_file_path = args.export_path
    extra_files = args.extra_files

    temp_files = []

    try:
        WorkflowExportUtils.validate_inputs(workflow_name, export_file_path)
        # Step 1 : Check if .war already exists with the given workflow name
        export_file_path = WorkflowExportUtils.check_war_already_exists(workflow_name, export_file_path, args.force)

        # Step 2 : Copy all artifacts to temp directory
        artifact_files = [workflow_spec_file, handler, extra_files]

        workflow_path = WorkflowExportUtils.copy_artifacts(workflow_name, artifact_files)

        # Step 2 : Zip 'em all up
        WorkflowExportUtils.archive(export_file_path, workflow_name, workflow_path, manifest)

        logging.info("Successfully exported workflow %s to file %s", workflow_name, export_file_path)
    except WorkflowArchiverError as e:
        logging.error(e)
        sys.exit(1)
    finally:
        WorkflowExportUtils.clean_temp_files(temp_files)


def generate_workflow_archive():
    """
    Generate a workflow archive file
    :return:
    """

    logging.basicConfig(format='%(levelname)s - %(message)s')
    args = ArgParser.workflow_archiver_args_parser().parse_args()
    manifest = WorkflowExportUtils.generate_manifest_json(args)
    package_workflow(args, manifest=manifest)


if __name__ == '__main__':
    generate_workflow_archive()
