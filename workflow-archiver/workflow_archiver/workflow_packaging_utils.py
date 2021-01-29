

"""
Helper utils for Workflow Archiver tool
"""

import logging
import os
import re
import zipfile
import shutil
import tempfile
from .workflow_archiver_error import WorkflowArchiverError

from .manifest_components.manifest import Manifest
from .manifest_components.workflow import Workflow

MANIFEST_FILE_NAME = 'MANIFEST.json'
WAR_INF = 'WAR-INF'


class WorkflowExportUtils(object):
    """
    Helper utils for Workflow Archiver tool.
    This class lists out all the methods such as validations for workflow archiving.
    """

    @staticmethod
    def get_archive_export_path(export_file_path, workflow_name):
        return os.path.join(export_file_path, f'{workflow_name}.war')

    @staticmethod
    def check_war_already_exists(workflow_name, export_file_path, overwrite):
        """
        Function to check if .war already exists
        :param workflow_name:
        :param export_file_path:
        :param overwrite:
        :return:
        """
        if export_file_path is None:
            export_file_path = os.getcwd()

        export_file = WorkflowExportUtils.get_archive_export_path(export_file_path, workflow_name)

        if os.path.exists(export_file):
            if overwrite:
                logging.warning("Overwriting %s ...", export_file)
            else:
                raise WorkflowArchiverError("{0} already exists.\n"
                                            "Please specify --force/-f option to overwrite the workflow archive "
                                            "output file.\n"
                                            "See -h/--help for more details.".format(export_file))

        return export_file_path

    @staticmethod
    def generate_workflow(workflow_args):
        workflow = Workflow(workflow_name=workflow_args.workflow_name, spec_file=workflow_args.spec_file,
                            handler=workflow_args.handler)
        return workflow

    @staticmethod
    def generate_manifest_json(args):
        """
        Function to generate manifest as a json string from the inputs provided by the user in the command line
        :param args:
        :return:s
        """

        workflow = WorkflowExportUtils.generate_workflow(args)

        manifest = Manifest(workflow=workflow)

        return str(manifest)

    @staticmethod
    def clean_temp_files(temp_files):
        for f in temp_files:
            os.remove(f)

    @staticmethod
    def make_dir(d):
        if not os.path.isdir(d):
            os.makedirs(d)

    @staticmethod
    def copy_artifacts(workflow_name, artifact_files):
        """
        copy workflow artifacts in a common workflow directory for archiving
        :param workflow_name: name of workflow being archived
        :param artifact_files: list of files to be copied in archive
        :return:
        """
        workflow_path = os.path.join(tempfile.gettempdir(), workflow_name)
        if os.path.exists(workflow_path):
            shutil.rmtree(workflow_path)
        WorkflowExportUtils.make_dir(workflow_path)
        for path in artifact_files:
            if path:
                for file in path.split(","):
                    shutil.copy(file, workflow_path)

        return workflow_path

    @staticmethod
    def archive(export_file, workflow_name, workflow_path, manifest):
        """
        Create a workflow-archive
        :param export_file:
        :param workflow_name:
        :param workflow_path
        :param manifest:
        :return:
        """
        war_path = WorkflowExportUtils.get_archive_export_path(export_file, workflow_name)
        try:
            with zipfile.ZipFile(war_path, 'w', zipfile.ZIP_DEFLATED) as z:
                WorkflowExportUtils.archive_dir(workflow_path, z)
                # Write the manifest here now as a json
                z.writestr(os.path.join(WAR_INF, MANIFEST_FILE_NAME), manifest)
        except IOError:
            logging.error("Failed to save the workflow-archive to workflow-path \"%s\". "
                          "Check the file permissions and retry.", export_file)
            raise
        except Exception as e:
            logging.error("Failed to convert %s to the workflow-archive.", workflow_name)
            raise e

    @staticmethod
    def archive_dir(path, dst):

        """
        This method zips the dir and filters out some files based on a expression
        :param path:
        :param dst:
        :return:
        """
        unwanted_dirs = {'__MACOSX', '__pycache__'}

        for root, directories, files in os.walk(path):
            # Filter directories
            directories[:] = [d for d in directories if WorkflowExportUtils.directory_filter(d, unwanted_dirs)]
            for f in files:
                file_path = os.path.join(root, f)
                dst.write(file_path, os.path.relpath(file_path, path))

    @staticmethod
    def directory_filter(directory, unwanted_dirs):
        """
        This method weeds out unwanted hidden directories from the workflow archive .war file
        :param directory:
        :param unwanted_dirs:
        :return:
        """
        if directory in unwanted_dirs:
            return False
        if directory.startswith('.'):
            return False

        return True

    @staticmethod
    def file_filter(current_file, files_to_exclude):
        """
        This method weeds out unwanted files
        :param current_file:
        :param files_to_exclude:
        :return:
        """
        files_to_exclude.add('MANIFEST.json')
        if current_file in files_to_exclude:
            return False

        elif current_file.endswith(('.pyc', '.DS_Store', '.war')):
            return False

        return True

    @staticmethod
    def check_workflow_name_regex_or_exit(workflow_name):
        """
        Method checks whether workflow name passes regex filter.
        If the regex Filter fails, the method exits.
        :param workflow_name:
        :return:
        """
        if not re.match(r'^[A-Za-z0-9][A-Za-z0-9_\-.]*$', workflow_name):
            raise WorkflowArchiverError("Workflow name contains special characters.\n"
                                        "The allowed regular expression filter for workflow "
                                        "name is: ^[A-Za-z0-9][A-Za-z0-9_\\-.]*$")

    @staticmethod
    def validate_inputs(workflow_name, export_path):
        WorkflowExportUtils.check_workflow_name_regex_or_exit(workflow_name)
        if not os.path.isdir(os.path.abspath(export_path)):
            raise WorkflowArchiverError("Given export-path {} is not a directory. "
                                        "Point to a valid export-path directory.".format(export_path))
