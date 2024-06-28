"""
Helper utils for Model Export tool
"""

import glob
import logging
import os
import re
import shutil
import sys
import tarfile
import tempfile
import zipfile
from io import BytesIO
from pathlib import Path

from .manifest_components.manifest import Manifest, RuntimeType
from .manifest_components.model import Model
from .model_archiver_config import ModelArchiverConfig
from .model_archiver_error import ModelArchiverError

archiving_options = {
    "tgz": ".tar.gz",
    "no-archive": "",
    "zip-store": ".mar",
    "default": ".mar",
}


model_handlers = {
    "base_handler": "anything",
    "text_classifier": "text",
    "image_classifier": "vision",
    "object_detector": "vision",
    "image_segmenter": "vision",
    "dali_image_classifier": "vision",
    "vllm_handler": "text",
}

MODEL_SERVER_VERSION = "1.0"
MODEL_ARCHIVE_VERSION = "1.0"
MANIFEST_FILE_NAME = "MANIFEST.json"
MAR_INF = "MAR-INF"


class ModelExportUtils(object):
    """
    Helper utils for Model Archiver tool.
    This class lists out all the methods such as validations for model archiving.
    """

    @staticmethod
    def get_archive_export_path(export_file_path, model_name, archive_format):
        return os.path.join(
            export_file_path,
            "{}{}".format(model_name, archiving_options.get(archive_format)),
        )

    @staticmethod
    def check_mar_already_exists(
        model_name, export_file_path, overwrite, archive_format="default"
    ):
        """
        Function to check if .mar already exists
        :param archive_format:
        :param model_name:
        :param export_file_path:
        :param overwrite:
        :return:
        """
        if export_file_path is None:
            export_file_path = os.getcwd()

        export_file = ModelExportUtils.get_archive_export_path(
            export_file_path, model_name, archive_format
        )

        if os.path.exists(export_file):
            if overwrite:
                logging.warning("Overwriting %s ...", export_file)
            else:
                raise ModelArchiverError(
                    "{0} already exists.\n"
                    "Please specify --force/-f option to overwrite the model archive "
                    "output file.\n"
                    "See -h/--help for more details.".format(export_file)
                )

        return export_file_path

    @staticmethod
    def find_unique(files, suffix):
        """
        Function to find unique model params file
        :param files:
        :param suffix:
        :return:
        """
        match = [f for f in files if f.endswith(suffix)]
        count = len(match)

        if count == 0:
            return None
        elif count == 1:
            return match[0]
        else:
            raise ModelArchiverError(
                "model-archiver expects only one {} file in the folder."
                " Found {} files {} in model-path.".format(suffix, count, match)
            )

    @staticmethod
    def generate_model(modelcfg: ModelArchiverConfig):
        model = Model(
            model_name=modelcfg.model_name,
            serialized_file=modelcfg.serialized_file,
            model_file=modelcfg.model_file,
            handler=modelcfg.handler,
            model_version=modelcfg.version,
            requirements_file=modelcfg.requirements_file,
            config_file=modelcfg.config_file,
        )
        return model

    @staticmethod
    def generate_manifest_json(config: ModelArchiverConfig) -> str:
        """
        Function to generate manifest as a json string from the inputs provided by the user in the command line
        :param args:
        :return:
        """

        model = ModelExportUtils.generate_model(config)

        manifest = Manifest(runtime=config.runtime, model=model)

        return str(manifest)

    @staticmethod
    def clean_temp_files(temp_files):
        for f in temp_files:
            os.remove(f)

    @staticmethod
    def make_dir(d):
        if not os.path.isdir(d):
            os.makedirs(d)

    @staticmethod
    def copy_artifacts(model_name, runtime, **kwargs):
        """
        copy model artifacts in a common model directory for archiving
        :param model_name: name of model being archived
        :param kwargs: key value pair of files to be copied in archive
        :return:
        """
        model_path = os.path.join(tempfile.gettempdir(), model_name)
        if os.path.exists(model_path):
            shutil.rmtree(model_path)
        ModelExportUtils.make_dir(model_path)
        for file_type, path in kwargs.items():
            if path:
                if file_type == "handler":
                    if (
                        runtime == RuntimeType.PYTHON.value
                        or runtime == RuntimeType.PYTHON3.value
                    ):
                        if path in model_handlers.keys():
                            continue

                        if ".py" not in path:
                            path = (path.split(":")[0] if ":" in path else path) + ".py"
                    elif runtime == RuntimeType.LSP.value:
                        if path in ["BaseHandler", "TorchScriptHandler"]:
                            continue

                        if sys.platform.startswith("linux"):
                            path = (path.split(":")[0] if ":" in path else path) + ".so"
                        elif sys.platform.startswith("darwin"):
                            path = (
                                path.split(":")[0] if ":" in path else path
                            ) + ".dylib"

                if file_type == "extra_files":
                    for path_or_wildcard in path.split(","):
                        if not Path(path_or_wildcard).exists():
                            raise FileNotFoundError(
                                f"File does not exist: {path_or_wildcard}"
                            )
                        for file in glob.glob(path_or_wildcard.strip()):
                            if os.path.isfile(file):
                                shutil.copy2(file, model_path)
                            elif os.path.isdir(file) and file != model_path:
                                for item in os.listdir(file):
                                    src = os.path.join(file, item)
                                    dst = os.path.join(model_path, item)
                                    if os.path.isfile(src):
                                        shutil.copy2(src, dst)
                                    elif os.path.isdir(src):
                                        shutil.copytree(src, dst, False, None)
                            else:
                                raise ValueError(f"Invalid extra file given {file}")
                else:
                    shutil.copy(path, model_path)

        return model_path

    @staticmethod
    def archive(
        export_file, model_name, model_path, manifest, archive_format="default"
    ):
        """
        Create a model-archive
        :param archive_format:
        :param export_file:
        :param model_name:
        :param model_path
        :param manifest:
        :return:
        """
        mar_path = ModelExportUtils.get_archive_export_path(
            export_file, model_name, archive_format
        )
        try:
            if archive_format == "tgz":
                with tarfile.open(mar_path, "w:gz") as z:
                    ModelExportUtils.archive_dir(
                        model_path, z, archive_format, model_name
                    )
                    # Write the manifest here now as a json
                    tar_manifest = tarfile.TarInfo(
                        name=os.path.join(model_name, MAR_INF, MANIFEST_FILE_NAME)
                    )
                    tar_manifest.size = len(manifest.encode("utf-8"))
                    z.addfile(tarinfo=tar_manifest, fileobj=BytesIO(manifest.encode()))
                    z.close()
            elif archive_format == "no-archive":
                if model_path != mar_path:
                    # Copy files to export path if
                    ModelExportUtils.archive_dir(
                        model_path, mar_path, archive_format, model_name
                    )
                # Write the MANIFEST in place
                manifest_path = os.path.join(mar_path, MAR_INF)
                ModelExportUtils.make_dir(manifest_path)
                with open(os.path.join(manifest_path, MANIFEST_FILE_NAME), "w") as f:
                    f.write(manifest)
            else:
                zip_mode = (
                    zipfile.ZIP_STORED
                    if archive_format == "zip-store"
                    else zipfile.ZIP_DEFLATED
                )
                with zipfile.ZipFile(mar_path, "w", zip_mode) as z:
                    ModelExportUtils.archive_dir(
                        model_path, z, archive_format, model_name
                    )
                    # Write the manifest here now as a json
                    z.writestr(os.path.join(MAR_INF, MANIFEST_FILE_NAME), manifest)
        except IOError:
            logging.error(
                'Failed to save the model-archive to model-path "%s". '
                "Check the file permissions and retry.",
                export_file,
            )
            raise
        except:
            logging.error("Failed to convert %s to the model-archive.", model_name)
            raise

    @staticmethod
    def archive_dir(path, dst, archive_format, model_name):
        """
        This method zips the dir and filters out some files based on a expression
        :param archive_format:
        :param path:
        :param dst:
        :param model_name:
        :return:
        """
        unwanted_dirs = {"__MACOSX", "__pycache__"}

        for root, directories, files in os.walk(path):
            # Filter directories
            directories[:] = [
                d
                for d in directories
                if ModelExportUtils.directory_filter(d, unwanted_dirs)
            ]
            for f in files:
                file_path = os.path.join(root, f)
                if archive_format == "tgz":
                    dst.add(
                        file_path,
                        arcname=os.path.join(
                            model_name, os.path.relpath(file_path, path)
                        ),
                    )
                elif archive_format == "no-archive":
                    dst_dir = os.path.dirname(
                        os.path.join(dst, os.path.relpath(file_path, path))
                    )
                    ModelExportUtils.make_dir(dst_dir)
                    shutil.copy(file_path, dst_dir)
                else:
                    dst.write(file_path, os.path.relpath(file_path, path))

    @staticmethod
    def directory_filter(directory, unwanted_dirs):
        """
        This method weeds out unwanted hidden directories from the model archive .mar file
        :param directory:
        :param unwanted_dirs:
        :return:
        """
        if directory in unwanted_dirs:
            return False
        if directory.startswith("."):
            return False

        return True

    @staticmethod
    def file_filter(current_file, files_to_exclude):
        """
        This method weeds out unwanted files
        :param current_file:
        :param files_to_exclude:
        :return:
        """
        files_to_exclude.add("MANIFEST.json")
        if current_file in files_to_exclude:
            return False

        elif current_file.endswith((".pyc", ".DS_Store", ".mar")):
            return False

        return True

    @staticmethod
    def check_model_name_regex_or_exit(model_name):
        """
        Method checks whether model name passes regex filter.
        If the regex Filter fails, the method exits.
        :param model_name:
        :return:
        """
        if not re.match(r"^[A-Za-z0-9][A-Za-z0-9_\-.]*$", model_name):
            raise ModelArchiverError(
                "Model name contains special characters.\n"
                "The allowed regular expression filter for model "
                "name is: ^[A-Za-z0-9][A-Za-z0-9_\\-.]*$"
            )

    @staticmethod
    def validate_inputs(model_name, export_path):
        ModelExportUtils.check_model_name_regex_or_exit(model_name)
        if not os.path.isdir(os.path.abspath(export_path)):
            raise ModelArchiverError(
                "Given export-path {} is not a directory. "
                "Point to a valid export-path directory.".format(export_path)
            )
