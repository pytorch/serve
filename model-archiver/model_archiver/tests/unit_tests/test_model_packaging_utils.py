import json
import platform
from collections import namedtuple
from pathlib import Path

import pytest
from model_archiver.manifest_components.manifest import RuntimeType
from model_archiver.model_archiver_error import ModelArchiverError
from model_archiver.model_packaging_utils import ModelExportUtils

MANIFEST_FILE = Path(__file__).parents[1].joinpath("integ_tests/MAR-INF/MANIFEST.json")


# noinspection PyClassHasNoInit
def _validate_mar(patches):
    if platform.system() == "Windows":
        patches.path_exists.assert_called_once_with("/Users/dummyUser\\some-model.mar")
    else:
        patches.path_exists.assert_called_once_with("/Users/dummyUser/some-model.mar")


# noinspection PyClassHasNoInit


class TestExportModelUtils:
    # noinspection PyClassHasNoInit
    class TestMarExistence:
        @pytest.fixture()
        def patches(self, mocker):
            Patches = namedtuple("Patches", ["getcwd", "path_exists"])
            patches = Patches(mocker.patch("os.getcwd"), mocker.patch("os.path.exists"))
            patches.getcwd.return_value = "/Users/dummyUser"
            return patches

        def test_export_file_is_none(self, patches):
            patches.path_exists.return_value = False
            ret_val = ModelExportUtils.check_mar_already_exists(
                "some-model", None, False
            )
            _validate_mar(patches)
            assert ret_val == "/Users/dummyUser"

        def test_export_file_is_not_none(self, patches):
            patches.path_exists.return_value = False
            ModelExportUtils.check_mar_already_exists(
                "some-model", "/Users/dummyUser/", False
            )
            patches.path_exists.assert_called_once_with(
                "/Users/dummyUser/some-model.mar"
            )

        def test_export_file_already_exists_with_override(self, patches):
            patches.path_exists.return_value = True
            ModelExportUtils.check_mar_already_exists("some-model", None, True)
            _validate_mar(patches)

        def test_export_file_already_exists_with_override_false(self, patches):
            patches.path_exists.return_value = True
            with pytest.raises(ModelArchiverError):
                ModelExportUtils.check_mar_already_exists("some-model", None, False)
            _validate_mar(patches)

        def test_export_file_is_none_tar(self, patches):
            patches.path_exists.return_value = False
            ret_val = ModelExportUtils.check_mar_already_exists(
                "some-model", None, False, archive_format="tgz"
            )

            if platform.system() == "Windows":
                patches.path_exists.assert_called_once_with(
                    "/Users/dummyUser\\some-model.tar.gz"
                )
            else:
                patches.path_exists.assert_called_once_with(
                    "/Users/dummyUser/some-model.tar.gz"
                )
            assert ret_val == "/Users/dummyUser"

    class TestArchiveTypes:
        def test_archive_types(self):
            from model_archiver.model_packaging_utils import (
                archiving_options as ar_opts,
            )

            assert ar_opts.get("tgz") == ".tar.gz"
            assert ar_opts.get("no-archive") == ""
            assert ar_opts.get("zip-store") == ".mar"
            assert ar_opts.get("default") == ".mar"
            assert len(ar_opts) == 4

    # noinspection PyClassHasNoInit
    class TestCustomModelTypes:
        model_path = "/Users/dummyUser"

        @pytest.fixture()
        def patches(self, mocker):
            Patches = namedtuple("Patches", ["utils", "listdir"])
            patch = Patches(
                mocker.patch("model_archiver.model_packaging_utils.ModelExportUtils"),
                mocker.patch("os.listdir"),
            )

            patch.listdir.return_value = {"a", "b", "c"}
            return patch

    # noinspection PyClassHasNoInit
    class TestFindUnique:
        def test_with_count_zero(self):
            files = ["a.txt", "b.txt", "c.txt"]
            suffix = ".mxnet"
            val = ModelExportUtils.find_unique(files, suffix)
            assert val is None

        def test_with_count_one(self):
            files = ["a.mxnet", "b.txt", "c.txt"]
            suffix = ".mxnet"
            val = ModelExportUtils.find_unique(files, suffix)
            assert val == "a.mxnet"

        def test_with_exit(self):
            files = ["a.onnx", "b.onnx", "c.txt"]
            suffix = ".onnx"
            with pytest.raises(ModelArchiverError):
                ModelExportUtils.find_unique(files, suffix)

    # noinspection PyClassHasNoInit
    class TestCleanTempFiles:
        @pytest.fixture()
        def patches(self, mocker):
            Patches = namedtuple("Patches", ["remove"])
            patches = Patches(mocker.patch("os.remove"))

            patches.remove.return_value = True
            return patches

        def test_clean_call(self, patches):
            temp_files = ["a", "b", "c"]
            ModelExportUtils.clean_temp_files(temp_files)

            patches.remove.assert_called()
            assert patches.remove.call_count == len(temp_files)

    # noinspection PyClassHasNoInit
    class TestGenerateManifestProps:
        class Namespace:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        model_name = "my-model"
        handler = "a.py::my-awesome-func"
        serialized_file = "model.pt"
        model_file = "model.pt"
        version = "1.0"
        requirements_file = "requirements.txt"

        args = Namespace(
            model_name=model_name,
            handler=handler,
            runtime=RuntimeType.PYTHON.value,
            serialized_file=serialized_file,
            model_file=model_file,
            version=version,
            requirements_file=requirements_file,
            config_file=None,
        )

        def test_model(self):
            mod = ModelExportUtils.generate_model(self.args)
            assert mod.model_name == self.model_name
            assert mod.handler == self.handler

        def test_manifest_json(self):
            manifest = ModelExportUtils.generate_manifest_json(self.args)
            manifest_json = json.loads(manifest)
            assert manifest_json["runtime"] == RuntimeType.PYTHON.value
            assert "model" in manifest_json
            assert "license" not in manifest_json

    # noinspection PyClassHasNoInit
    class TestModelNameRegEx:
        def test_regex_pass(self):
            model_names = [
                "my-awesome-model",
                "Aa.model",
                "a",
                "aA.model",
                "a1234.model",
                "a-A-A.model",
                "123-abc",
            ]
            for m in model_names:
                ModelExportUtils.check_model_name_regex_or_exit(m)

        def test_regex_fail(self):
            model_names = [
                "abc%",
                "123$abc",
                "abc!123",
                "@123",
                "(model",
                "mdoel)",
                "12*model-a.model",
                "##.model",
                "-.model",
            ]
            for m in model_names:
                with pytest.raises(ModelArchiverError):
                    ModelExportUtils.check_model_name_regex_or_exit(m)

    # noinspection PyClassHasNoInit
    class TestFileFilter:
        files_to_exclude = {"abc.onnx"}

        def test_with_return_false(self):
            assert (
                ModelExportUtils.file_filter("abc.onnx", self.files_to_exclude) is False
            )

        def test_with_pyc(self):
            assert (
                ModelExportUtils.file_filter("abc.pyc", self.files_to_exclude) is False
            )

        def test_with_ds_store(self):
            assert (
                ModelExportUtils.file_filter(".DS_Store", self.files_to_exclude)
                is False
            )

        def test_with_return_true(self):
            assert (
                ModelExportUtils.file_filter("abc.mxnet", self.files_to_exclude) is True
            )

    # noinspection PyClassHasNoInit
    class TestDirectoryFilter:
        unwanted_dirs = {"__MACOSX", "__pycache__"}

        def test_with_unwanted_dirs(self):
            assert (
                ModelExportUtils.directory_filter("__MACOSX", self.unwanted_dirs)
                is False
            )

        def test_with_starts_with_dot(self):
            assert (
                ModelExportUtils.directory_filter(".gitignore", self.unwanted_dirs)
                is False
            )

        def test_with_return_true(self):
            assert (
                ModelExportUtils.directory_filter("my-model", self.unwanted_dirs)
                is True
            )


def create_manifest_from_test_json(test_json):
    test_ = {k.replace("-", "_"): v for k, v in test_json.items()}
    test_["requirements_file"] = ""
    test_["runtime"] = RuntimeType.PYTHON3.value
    test_["config_file"] = ""

    args = namedtuple("Model", test_.keys())(**test_)
    manifest = ModelExportUtils.generate_manifest_json(args)
    return manifest


def prepare_model_dir(test_name, integ_tests):
    integ_tests = list(filter(lambda t: t["name"] == test_name, integ_tests))
    assert len(integ_tests) == 1
    test = integ_tests[0]

    keys = (
        "model-file",
        "serialized-file",
        "handler",
        "extra-files",
        "config-file",
    )
    artifact_files = {k.replace("-", "_"): test[k] for k in keys}

    model_path = ModelExportUtils.copy_artifacts(test["model-name"], **artifact_files)

    manifest = create_manifest_from_test_json(test)
    return manifest, model_path


def test_archive_creation_with_zip_store(tmp_path, integ_tests, mocker):
    manifest, model_path = prepare_model_dir("packaging_zip_store_mar", integ_tests)

    ModelExportUtils.archive(
        tmp_path,
        "zip-store",
        model_path,
        manifest,
        archive_format="zip-store",
    )

    ModelExportUtils.archive(
        tmp_path, "zip", model_path, manifest, archive_format="default"
    )

    stored_size = Path(tmp_path).joinpath("zip-store.mar").stat().st_size
    zipped_size = Path(tmp_path).joinpath("zip.mar").stat().st_size

    assert zipped_size < stored_size


def test_missing_extra_files(integ_tests):
    with pytest.raises(FileNotFoundError):
        prepare_model_dir("missing_extra_files", integ_tests)
