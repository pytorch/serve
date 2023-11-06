import os
from argparse import Namespace
from dataclasses import dataclass, fields
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

    @classmethod
    def from_args(cls, args: Namespace) -> "ModelArchiverConfig":
        params = {field.name: getattr(args, field.name) for field in fields(cls)}
        config = cls(**params)
        return config
