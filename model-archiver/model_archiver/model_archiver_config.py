import os
from argparse import Namespace
from dataclasses import dataclass
from typing import Literal, Optional

from model_archiver.manifest_components.manifest import RuntimeType


@dataclass
class ModelArchiverConfig:
    model_name: str
    handler: str
    version: str
    serialized_file: Optional[str] = None
    model_file: Optional[str] = None
    extra_files: Optional[str] = None
    runtime: str = RuntimeType.PYTHON.value
    export_path: str = os.getcwd()
    archive_format: Literal["default", "tgz", "no-archive"] = "default"
    force: bool = False
    requirements_file: Optional[str] = None
    config_file: Optional[str] = None

    @staticmethod
    def from_args(args: Namespace) -> "ModelArchiverConfig":
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
