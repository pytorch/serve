

"""
Command line interface to export model files to be used for inference by MXNet Model Server
"""

import logging
import sys
from .arg_parser import ArgParser
from .model_packaging_utils import ModelExportUtils
from .model_archiver_error import ModelArchiverError


def package_model(args, manifest):
    """
    Internal helper for the exporting model command line interface.
    """
    model_path = args.model_path
    model_name = args.model_name
    export_file_path = args.export_path
    temp_files = []

    try:
        ModelExportUtils.validate_inputs(model_path, model_name, export_file_path)
        # Step 1 : Check if .mar already exists with the given model name
        export_file_path = ModelExportUtils.check_mar_already_exists(model_name, export_file_path,
                                                                     args.force, args.archive_format)

        # Step 2 : Check if any special handling is required for custom models like onnx models
        files_to_exclude = []
        if args.convert:
            t, files_to_exclude = ModelExportUtils.check_custom_model_types(model_path, model_name)
            temp_files.extend(t)

        # Step 3 : Zip 'em all up
        ModelExportUtils.archive(export_file_path, model_name, model_path, files_to_exclude, manifest,
                                 args.archive_format)

        logging.info("Successfully exported model %s to file %s", model_name, export_file_path)
    except ModelArchiverError as e:
        logging.error(e)
        sys.exit(1)
    finally:
        ModelExportUtils.clean_temp_files(temp_files)


def generate_model_archive():
    """
    Generate a model archive file
    :return:
    """
    logging.basicConfig(format='%(levelname)s - %(message)s')
    args = ArgParser.export_model_args_parser().parse_args()
    manifest = ModelExportUtils.generate_manifest_json(args)
    package_model(args, manifest=manifest)


if __name__ == '__main__':
    generate_model_archive()
