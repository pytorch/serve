from argparse import Namespace
from collections import namedtuple

import pytest
from model_archiver import ModelArchiver, ModelArchiverConfig
from model_archiver.manifest_components.manifest import RuntimeType


# noinspection PyClassHasNoInit
class TestModelArchiver:
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
            mocker.patch("model_archiver.arg_parser.ArgParser"),
            mocker.patch("model_archiver.model_packaging.ModelExportUtils"),
            mocker.patch("model_archiver.model_packaging.package_model"),
        )
        mocker.patch("shutil.rmtree")

        return patches

    def test_gen_model_archive(self, patches):
        ModelArchiver.generate_model_archive(self.config)
        patches.export_method.assert_called()

    def test_model_archiver_config_from_args(self):
        args = Namespace(
            model_name=self.model_name,
            handler=self.handler,
            runtime=RuntimeType.PYTHON.value,
            model_file=self.model_file,
            serialized_file=self.serialized_file,
            extra_files=None,
            export_path=self.export_path,
            force=False,
            archive_format="default",
            version=self.version,
            requirements_file=self.requirements_file,
            config_file=None,
        )
        config = ModelArchiverConfig.from_args(args)

        config == self.config
