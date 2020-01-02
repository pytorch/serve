
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
    model_file = args.model_file
    serialized_file = args.serialized_file
    model_name = args.model_name
    handler = args.handler
    extra_files = args.extra_files
    export_file_path = args.export_path
    temp_files = []

    try:
        ModelExportUtils.validate_inputs(model_name, export_file_path)
        # Step 1 : Check if .mar already exists with the given model name
        export_file_path = ModelExportUtils.check_mar_already_exists(model_name, export_file_path,
                                                                     args.force, args.archive_format)

        # Step 2 : Copy all artifacts for temp directory
        artifact_files = {'model_file': model_file, 'serialized_file': serialized_file, 'handler': handler,
                          'extra_files': extra_files}

        model_path = ModelExportUtils.copy_artifacts(model_name, **artifact_files)

        # Step 2 : Zip 'em all up
        ModelExportUtils.archive(export_file_path, model_name, model_path, manifest, args.archive_format)

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
    model_handlers = {
        'text': ['text_classifier', 'language_translator'],
        'vision': ['image_classifier', 'object_detector'],
        'audio': []
    }

    requires_destination_vocab = ['language_translator']

    logging.basicConfig(format='%(levelname)s - %(message)s')
    args = ArgParser.export_model_args_parser().parse_args()

    args_dict = vars(args)

    subtype_not_present = 'model_sub_type' not in args_dict or \
                          args_dict['model_sub_type'] not in model_handlers[args_dict['model_type']]

    if args.model_type in model_handlers.keys():
        if subtype_not_present and 'handler' not in args_dict:
            raise Exception("Unsupported model subtype for {0} models. Can be one of {1}. Or provide a custom handler"
                            .format(args['model_type'], str(model_handlers[args['model_type']])))

        if args.model_sub_type in requires_destination_vocab and 'destination_vocab' not in vars(args):
            raise Exception("Please provide the destination language vocab for {0} model.".format(args.model_sub_type))

    manifest = ModelExportUtils.generate_manifest_json(args)
    package_model(args, manifest=manifest)


if __name__ == '__main__':
    generate_model_archive()
