

"""
This module parses the arguments given through the torch-workflow-archiver command-line.
"""

import argparse
import os


# noinspection PyTypeChecker
class ArgParser(object):

    """
    Argument parser for torch-workflow-archiver commands
    """

    @staticmethod
    def workflow_archiver_args_parser():

        """ Argument parser for torch-workflow-archiver
        """

        parser_workflow_archiver = argparse.ArgumentParser(prog='torch-workflow-archiver',
                                                           description='Torch Workflow Archiver Tool',
                                                           formatter_class=argparse.RawTextHelpFormatter)

        parser_workflow_archiver.add_argument('--workflow-name',
                                              required=True,
                                              type=str,
                                              default=None,
                                              help='Exported workflow name. Exported file will be named as'
                                                   ' workflow-name.war and saved in current working directory '
                                                   'if no --export-path is specified, '
                                                   'else it will be saved under the export path')

        parser_workflow_archiver.add_argument('--spec-file',
                                              required=True,
                                              type=str,
                                              default=None,
                                              help='Path to .yaml file containing workflow DAG specification.')

        parser_workflow_archiver.add_argument('--handler',
                                              required=False,
                                              dest="handler",
                                              type=str,
                                              default=None,
                                              help="Path to python file containing workflow's "
                                                   "pre-process and post-process logic.")

        parser_workflow_archiver.add_argument('--export-path',
                                              required=False,
                                              type=str,
                                              default=os.getcwd(),
                                              help='Path where the exported .war file will be saved.'
                                                   'This is an optional parameter. If --export-path is not specified,'
                                                   ' the file will be saved in the current working directory. ')

        parser_workflow_archiver.add_argument('-f', '--force',
                                              required=False,
                                              action='store_true',
                                              help='When the -f or --force flag is specified, an existing .war file'
                                                   ' with same name as that provided in --workflow-name in the path'
                                                   ' specified by --export-path will be overwritten')

        parser_workflow_archiver.add_argument('--extra-files',
                                              required=False,
                                              type=str,
                                              default=None,
                                              help='Comma separated path to extra dependency files.')

        return parser_workflow_archiver
