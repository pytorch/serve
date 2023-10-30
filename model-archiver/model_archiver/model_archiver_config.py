import os
from argparse import Namespace
from typing import Literal, Optional

from model_archiver.manifest_components.manifest import RuntimeType


class ModelArchiverConfig:
    def __init__(
        self,
        model_name: str,
        handler: str,
        version: str,
        serialized_file: Optional[str] = None,
        model_file: Optional[str] = None,
        extra_files: Optional[str] = None,
        runtime: str = RuntimeType.PYTHON.value,
        export_path: str = os.getcwd(),
        archive_format: Literal["default", "tgz", "no-archive"] = "default",
        force: bool = False,
        requirements_file: Optional[str] = None,
        config_file: Optional[str] = None,
    ):
        self.model_name = model_name
        self.handler = handler
        self.version = version
        self.serialized_file = serialized_file
        self.model_file = model_file
        self.extra_files = extra_files
        self.runtime = runtime
        self.export_path = export_path
        self.archive_format = archive_format
        self.force = force
        self.requirements_file = requirements_file
        self.config_file = config_file

    @staticmethod
    def from_args(args: Optional[Namespace] = None) -> "ModelArchiverConfig":
        if args is None:
            from model_archiver.arg_parser import ArgParser

            args = ArgParser.export_model_args_parser().parse_args()

        config = ModelArchiverConfig(
            model_name=args.model_name,
            handler=args.handler,
            version=args.version,
            serialized_file=args.serialized_file,
            model_file=args.model_file,
            extra_files=args.extra_files,
            runtime=args.runtime,
            export_path=args.export_path,
            archive_format=args.archive_format,
            force=args.force,
            requirements_file=args.requirements_file,
            config_file=args.config_file,
        )
        return config
