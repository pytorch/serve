# Copyright 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#     http://www.apache.org/licenses/LICENSE-2.0
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

"""
This module parses the arguments given through the mxnet-model-server command-line. This is used by model-server
at runtime.
"""

import argparse
import os
from .manifest_components.manifest import RuntimeType


# noinspection PyTypeChecker
class ArgParser(object):

    """
    Argument parser for model-export-tool commands
    More detailed example is available at https://github.com/awslabs/mxnet-model-server/blob/master/README.md
    """

    @staticmethod
    def export_model_args_parser():

        """ Argument parser for mxnet-model-export
        """

        parser_export = argparse.ArgumentParser(prog='model-archiver', description='Model Archiver Tool',
                                                formatter_class=argparse.RawTextHelpFormatter)

        parser_export.add_argument('--model-name',
                                   required=True,
                                   type=str,
                                   default=None,
                                   help='Exported model name. Exported file will be named as\n'
                                        'model-name.mar and saved in current working directory if no --export-path is\n'
                                        'specified, else it will be saved under the export path')

        parser_export.add_argument('--model-path',
                                   required=True,
                                   type=str,
                                   default=None,
                                   help='Path to the folder containing model related files.')

        parser_export.add_argument('--handler',
                                   required=True,
                                   dest="handler",
                                   type=str,
                                   default=None,
                                   help='Handler path to handle custom MMS inference logic.')

        parser_export.add_argument('--runtime',
                                   required=False,
                                   type=str,
                                   default=RuntimeType.PYTHON.value,
                                   choices=[s.value for s in RuntimeType],
                                   help='The runtime specifies which language to run your inference code on.\n'
                                        'The default runtime is "python".')

        parser_export.add_argument('--export-path',
                                   required=False,
                                   type=str,
                                   default=os.getcwd(),
                                   help='Path where the exported .mar file will be saved. This is an optional\n'
                                        'parameter. If --export-path is not specified, the file will be saved in the\n'
                                        'current working directory. ')

        parser_export.add_argument('--archive-format',
                                   required=False,
                                   type=str,
                                   default="default",
                                   choices=["tgz", "no-archive", "default"],
                                   help='The format in which the model artifacts are archived.\n'
                                        '"tgz": This creates the model-archive in <model-name>.tar.gz format.\n'
                                        'If platform hosting MMS requires model-artifacts to be in ".tar.gz"\n'
                                        'use this option.\n'
                                        '"no-archive": This option creates an non-archived version of model artifacts\n'
                                        'at "export-path/{model-name}" location. As a result of this choice, \n'
                                        'MANIFEST file will be created at "export-path/{model-name}" location\n'
                                        'without archiving these model files\n'
                                        '"default": This creates the model-archive in <model-name>.mar format.\n'
                                        'This is the default archiving format. Models archived in this format\n'
                                        'will be readily hostable on native MMS.\n')

        parser_export.add_argument('-f', '--force',
                                   required=False,
                                   action='store_true',
                                   help='When the -f or --force flag is specified, an existing .mar file with same\n'
                                        'name as that provided in --model-name in the path specified by --export-path\n'
                                        'will overwritten')

        parser_export.add_argument('-c', '--convert',
                                   required=False,
                                   action='store_true',
                                   help='When this option is used, model-archiver looks for special files and tries\n'
                                        'preprocesses them. For example, if this option is chosen when running\n'
                                        'model-archiver tool on a model with ".onnx" extension, the tool will try and\n'
                                        'convert ".onnx" model into an MXNet model.')


        return parser_export
