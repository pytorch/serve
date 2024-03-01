"""
Command line interface to export model files to be used for inference by MXNet Model Server
"""

import logging
import shutil
from typing import Optional

from model_archiver.arg_parser import ArgParser
from model_archiver.model_archiver_config import ModelArchiverConfig
from model_archiver.model_archiver_error import ModelArchiverError
from model_archiver.model_packaging_utils import ModelExportUtils


def package_model(config: ModelArchiverConfig, manifest: str):
    """
    Internal helper for the exporting model command line interface.
    """
    model_file = config.model_file
    serialized_file = config.serialized_file
    model_name = config.model_name
    handler = config.handler
    extra_files = config.extra_files
    export_file_path = config.export_path
    requirements_file = config.requirements_file
    config_file = config.config_file

    try:
        ModelExportUtils.validate_inputs(model_name, export_file_path)
        # Step 1 : Check if .mar already exists with the given model name
        export_file_path = ModelExportUtils.check_mar_already_exists(
            model_name, export_file_path, config.force, config.archive_format
        )

        # Step 2 : Copy all artifacts to temp directory
        artifact_files = {
            "model_file": model_file,
            "serialized_file": serialized_file,
            "handler": handler,
            "extra_files": extra_files,
            "requirements-file": requirements_file,
            "config_file": config_file,
        }

        model_path = ModelExportUtils.copy_artifacts(
            model_name, config.runtime, **artifact_files
        )

        # Step 2 : Zip 'em all up
        ModelExportUtils.archive(
            export_file_path, model_name, model_path, manifest, config.archive_format
        )
        shutil.rmtree(model_path)
        logging.info(
            "Successfully exported model %s to file %s", model_name, export_file_path
        )
    except ModelArchiverError as e:
        logging.error(e)
        raise e


def generate_model_archive(config: Optional[ModelArchiverConfig] = None):
    """
    Generate a model archive file
    :return:
    """

    logging.basicConfig(format="%(levelname)s - %(message)s")
    if config is None:
        config = ArgParser.export_model_args_parser()
    manifest = ModelExportUtils.generate_manifest_json(config)
    package_model(config, manifest=manifest)


if __name__ == "__main__":
    generate_model_archive()
