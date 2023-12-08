"""
This module parses the arguments given through the torch-model-archiver command-line.
"""

import argparse
import os

from .manifest_components.manifest import RuntimeType
from .model_archiver_config import ModelArchiverConfig


# noinspection PyTypeChecker
class ArgParser(object):

    """
    Argument parser for torch-model-archiver commands
    """

    @staticmethod
    def export_model_args_parser():
        """Argument parser for torch-model-export"""

        parser_export = argparse.ArgumentParser(
            prog="torch-model-archiver",
            description="Torch Model Archiver Tool",
            formatter_class=argparse.RawTextHelpFormatter,
        )

        parser_export.add_argument(
            "--model-name",
            required=True,
            type=str,
            default=None,
            help="Exported model name. Exported file will be named as\n"
            "model-name.mar and saved in current working directory if no --export-path is\n"
            "specified, else it will be saved under the export path",
        )

        parser_export.add_argument(
            "--serialized-file",
            required=False,
            type=str,
            default=None,
            help="Path to .pt or .pth file containing state_dict in case of eager mode\n"
            "or an executable ScriptModule in case of TorchScript or TensorRT\n"
            "or a .onnx file in the case of ORT.",
        )

        parser_export.add_argument(
            "--model-file",
            required=False,
            type=str,
            default=None,
            help="Path to python file containing model architecture.\n"
            "This parameter is mandatory for eager mode models.\n"
            "The model architecture file must contain only one\n"
            "class definition extended from torch.nn.Module.",
        )

        parser_export.add_argument(
            "--handler",
            required=True,
            dest="handler",
            type=str,
            default=None,
            help="TorchServe's default handler name\n"
            " or Handler path to handle custom inference logic.",
        )

        parser_export.add_argument(
            "--extra-files",
            required=False,
            type=str,
            default=None,
            help="Comma separated path to extra dependency files.",
        )

        parser_export.add_argument(
            "--runtime",
            required=False,
            type=str,
            default=RuntimeType.PYTHON.value,
            choices=[s.value for s in RuntimeType],
            help="The runtime specifies which language to run your inference code on.\n"
            'The default runtime is "python".',
        )

        parser_export.add_argument(
            "--export-path",
            required=False,
            type=str,
            default=os.getcwd(),
            help="Path where the exported .mar file will be saved. This is an optional\n"
            "parameter. If --export-path is not specified, the file will be saved in the\n"
            "current working directory. ",
        )

        parser_export.add_argument(
            "--archive-format",
            required=False,
            type=str,
            default="default",
            choices=["tgz", "no-archive", "zip-store", "default"],
            help="The format in which the model artifacts are archived.\n"
            '"tgz": This creates the model-archive in <model-name>.tar.gz format.\n'
            'If platform hosting TorchServe requires model-artifacts to be in ".tar.gz"\n'
            "use this option.\n"
            '"no-archive": This option creates an non-archived version of model artifacts\n'
            'at "export-path/{model-name}" location. As a result of this choice, \n'
            'MANIFEST file will be created at "export-path/{model-name}" location\n'
            "without archiving these model files\n"
            '"zip-store": This creates the model-archive in <model-name>.mar format\n'
            "but will skip deflating the files to speed up creation. Mainly used\n"
            "for testing purposes\n"
            '"default": This creates the model-archive in <model-name>.mar format.\n'
            "This is the default archiving format. Models archived in this format\n"
            "will be readily hostable on native TorchServe.\n",
        )

        parser_export.add_argument(
            "-f",
            "--force",
            required=False,
            action="store_true",
            help="When the -f or --force flag is specified, an existing .mar file with same\n"
            "name as that provided in --model-name in the path specified by --export-path\n"
            "will overwritten",
        )

        parser_export.add_argument(
            "-v",
            "--version",
            required=True,
            type=str,
            default=None,
            help="Model's version",
        )

        parser_export.add_argument(
            "-r",
            "--requirements-file",
            required=False,
            type=str,
            default=None,
            help="Path to a requirements.txt containing model specific python dependency\n"
            " packages.",
        )

        parser_export.add_argument(
            "-c",
            "--config-file",
            required=False,
            type=str,
            default=None,
            help="Path to a yaml file containing model configuration eg. batch_size.",
        )

        return ModelArchiverConfig.from_args(parser_export.parse_args())
