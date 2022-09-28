"""
Command line interface to export model files to be used for inference by MXNet Model Server
"""

import logging
import shutil
import sys

from model_archiver.arg_parser import ArgParser
from model_archiver.model_archiver_error import ModelArchiverError
from model_archiver.model_packaging_utils import ModelExportUtils


def package_model(args, manifest):
    """
    Internal helper for the exporting model command line interface.
    """
    model_file = args.model_file
    serialized_file = args.serialized_file
    model_name = args.model_name
    handler = args.handler
    extra_files = args.extra_files
    export_file_path = args.export_path
    requirements_file = args.requirements_file

    try:
        ModelExportUtils.validate_inputs(model_name, export_file_path)
        # Step 1 : Check if .mar already exists with the given model name
        export_file_path = ModelExportUtils.check_mar_already_exists(
            model_name, export_file_path, args.force, args.archive_format
        )

        # Step 2 : Copy all artifacts to temp directory
        artifact_files = {
            "model_file": model_file,
            "serialized_file": serialized_file,
            "handler": handler,
            "extra_files": extra_files,
            "requirements-file": requirements_file,
        }

        model_path = ModelExportUtils.copy_artifacts(model_name, **artifact_files)

        # Step 2 : Zip 'em all up
        ModelExportUtils.archive(
            export_file_path, model_name, model_path, manifest, args.archive_format
        )
        shutil.rmtree(model_path)
        logging.info(
            "Successfully exported model %s to file %s", model_name, export_file_path
        )
    except ModelArchiverError as e:
        logging.error(e)
        sys.exit(1)


def generate_model_archive():
    """
    Generate a model archive file
    :return:
    """

    logging.basicConfig(format="%(levelname)s - %(message)s")
    args = ArgParser.export_model_args_parser().parse_args()
    manifest = ModelExportUtils.generate_manifest_json(args)
    package_model(args, manifest=manifest)


if __name__ == "__main__":
    generate_model_archive()
