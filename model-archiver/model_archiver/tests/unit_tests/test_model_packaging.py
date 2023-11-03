from collections import namedtuple

import pytest
from model_archiver import ModelArchiverConfig
from model_archiver.manifest_components.manifest import RuntimeType
from model_archiver.model_packaging import generate_model_archive, package_model
from model_archiver.model_packaging_utils import ModelExportUtils


# noinspection PyClassHasNoInit
class TestModelPackaging:
    model_name = "my-model"
    model_file = "my-model/"
    serialized_file = "my-model/"
    handler = "a.py::my-awesome-func"
    export_path = "/Users/dummyUser/"
    version = "1.0"
    requirements_file = "requirements.txt"
    config_file = None

    config = ModelArchiverConfig(
        model_name=model_name,
        handler=handler,
        runtime=RuntimeType.PYTHON.value,
        model_file=model_file,
        serialized_file=serialized_file,
        extra_files=None,
        export_path=export_path,
        force=False,
        archive_format="default",
        version=version,
        requirements_file=requirements_file,
        config_file=None,
    )

    @pytest.fixture()
    def patches(self, mocker):
        Patches = namedtuple("Patches", ["arg_parse", "export_utils", "export_method"])
        patches = Patches(
            mocker.patch("model_archiver.model_packaging.ArgParser"),
            mocker.patch("model_archiver.model_packaging.ModelExportUtils"),
            mocker.patch("model_archiver.model_packaging.package_model"),
        )
        mocker.patch("shutil.rmtree")

        return patches

    def test_gen_model_archive(self, patches):
        patches.arg_parse.export_model_args_parser.parse_args.return_value = self.config
        generate_model_archive()
        patches.export_method.assert_called()

    def test_export_model_method(self, patches):
        patches.export_utils.check_mar_already_exists.return_value = "/Users/dummyUser/"
        patches.export_utils.check_custom_model_types.return_value = (
            "/Users/dummyUser",
            ["a.txt", "b.txt"],
        )
        patches.export_utils.zip.return_value = None

        package_model(self.config, ModelExportUtils.generate_manifest_json(self.config))
        patches.export_utils.validate_inputs.assert_called()
        patches.export_utils.archive.assert_called()

    def test_export_model_method_tar(self, patches):
        self.config.archive_format = "tgz"
        patches.export_utils.check_mar_already_exists.return_value = "/Users/dummyUser/"
        patches.export_utils.check_custom_model_types.return_value = (
            "/Users/dummyUser",
            ["a.txt", "b.txt"],
        )

        package_model(self.config, ModelExportUtils.generate_manifest_json(self.config))
        patches.export_utils.validate_inputs.assert_called()
        patches.export_utils.archive.assert_called()

    def test_export_model_method_noarchive(self, patches):
        self.config.archive_format = "no-archive"
        patches.export_utils.check_mar_already_exists.return_value = "/Users/dummyUser/"
        patches.export_utils.check_custom_model_types.return_value = (
            "/Users/dummyUser",
            ["a.txt", "b.txt"],
        )
        patches.export_utils.zip.return_value = None

        package_model(self.config, ModelExportUtils.generate_manifest_json(self.config))
        patches.export_utils.validate_inputs.assert_called()
        patches.export_utils.archive.assert_called()
