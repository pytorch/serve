import errno
import json
import os
import platform
import shutil
import tempfile
import time
from argparse import Namespace
from datetime import datetime
from pathlib import Path

import model_archiver

DEFAULT_RUNTIME = "python"
MANIFEST_FILE = "MAR-INF/MANIFEST.json"
INTEG_TEST_CONFIG_FILE = "integ_tests/configuration.json"
DEFAULT_HANDLER_CONFIG_FILE = "integ_tests/default_handler_configuration.json"

TEST_ROOT_DIR = Path(__file__).parents[1]
MODEL_ARCHIVER_ROOT_DIR = Path(__file__).parents[3]


def create_file_path(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def delete_file_path(path):
    try:
        if os.path.isfile(path):
            os.remove(path)
        if os.path.isdir(path):
            shutil.rmtree(path)
    except OSError:
        pass


def run_test(test, args, mocker):
    m = mocker.patch(
        "model_archiver.model_packaging.ArgParser.export_model_args_parser",
    )
    m.return_value.parse_args.return_value = args
    mocker.patch("sys.exit", side_effect=Exception())
    from model_archiver.model_packaging import generate_model_archive

    it = test.get("iterations", 1)
    for i in range(it):
        try:
            generate_model_archive()
        except Exception as exc:
            if test.get("expect-error") is not True:
                assert 0, str(exc)
            else:
                return 0
    # In case we expect an error we should not be here
    if test.get("expect-error") is True:
        assert 0, f"Error expected in test: {test['name']}"
    return 1


def validate_archive_exists(test):
    fmt = test.get("archive-format")
    if fmt == "tgz":
        assert os.path.isfile(
            os.path.join(test.get("export-path"), test.get("model-name") + ".tar.gz")
        )
    elif fmt == "no-archive":
        assert os.path.isdir(
            os.path.join(test.get("export-path"), test.get("model-name"))
        )
    else:
        assert os.path.isfile(
            os.path.join(test.get("export-path"), test.get("model-name") + ".mar")
        )


def validate_manifest_file(manifest, test, default_handler=None):
    """
    Validate the MANIFEST file
    :param manifest:
    :param test:
    :return:
    """
    assert datetime.strptime(manifest.get("createdOn"), "%d/%m/%Y %H:%M:%S")
    assert manifest.get("runtime") == test.get("runtime")
    assert manifest.get("model").get("modelName") == test.get("model-name")
    if not default_handler:
        assert (
            manifest.get("model").get("handler") == test.get("handler").split("/")[-1]
        )
    else:
        assert manifest.get("model").get("handler") == test.get("handler")
    assert manifest.get("archiverVersion") == model_archiver.__version__


def validate_files(file_list, prefix, default_handler=None):
    assert os.path.join(prefix, MANIFEST_FILE) in file_list
    assert os.path.join(prefix, "test_handler.py") in file_list
    assert os.path.join(prefix, "test_model.py") in file_list
    assert os.path.join(prefix, "test_serialized_file.pt") in file_list
    assert os.path.join(prefix, "dummy-artifacts.txt") in file_list
    assert os.path.join(prefix, "1.py") in file_list

    if default_handler == "text_classifier":
        assert os.path.join(prefix, "source_vocab.pt") in file_list


def validate_tar_archive(test_cfg):
    import tarfile

    file_name = os.path.join(
        test_cfg.get("export-path"), test_cfg.get("model-name") + ".tar.gz"
    )
    f = tarfile.open(file_name, "r:gz")
    manifest = json.loads(
        f.extractfile(os.path.join(test_cfg.get("model-name"), MANIFEST_FILE)).read()
    )
    validate_manifest_file(manifest, test_cfg)
    validate_files(f.getnames(), test_cfg.get("model-name"))


def validate_noarchive_archive(test):
    file_name = os.path.join(
        test.get("export-path"), test.get("model-name"), MANIFEST_FILE
    )
    manifest = json.loads(open(file_name).read())
    validate_manifest_file(manifest, test)


def validate_mar_archive(test):
    import zipfile

    file_name = os.path.join(test.get("export-path"), test.get("model-name") + ".mar")
    zf = zipfile.ZipFile(file_name, "r")
    manifest = json.loads(zf.open(MANIFEST_FILE).read())
    validate_manifest_file(manifest, test)


def validate_archive_content(test):
    fmt = test.get("archive-format")
    if fmt == "tgz":
        validate_tar_archive(test)
    if fmt == "no-archive":
        validate_noarchive_archive(test)
    if fmt == "default":
        validate_mar_archive(test)


def validate(test):
    validate_archive_exists(test)
    validate_archive_content(test)


def build_namespace(test):
    keys = [
        "model-name",
        "model-file",
        "serialized-file",
        "handler",
        "extra-files",
        "archive-format",
        "version",
        "export-path",
        "runtime",
        "requirements-file",
        "config-file",
        "force",
    ]
    test["requirements-file"] = None
    test["config-file"] = None
    test["force"] = test.get("force", False)
    test["runtime"] = test.get("runtime", DEFAULT_RUNTIME)
    test["archive-format"] = test.get("archive-format", "default")

    args = Namespace(**{k.replace("-", "_"): test[k] for k in keys})

    return args


def make_paths_absolute(test, keys):
    def make_absolute(paths):
        if "," in paths:
            return ",".join([make_absolute(p) for p in paths.split(",")])
        return MODEL_ARCHIVER_ROOT_DIR.joinpath(paths).as_posix()

    for k in keys:
        test[k] = make_absolute(test[k])

    return test


def test_model_archiver(integ_tests, mocker):
    for test in integ_tests:
        # tar.gz format problem on windows hence ignore
        if platform.system() == "Windows" and test["archive-format"] == "tgz":
            continue
        try:
            test["export-path"] = os.path.join(
                tempfile.gettempdir(), test["export-path"]
            )
            delete_file_path(test.get("export-path"))
            create_file_path(test.get("export-path"))
            test["runtime"] = test.get("runtime", DEFAULT_RUNTIME)
            test["model-name"] = (
                test["model-name"] + "_" + str(int(time.time() * 1000.0))
            )
            args = build_namespace(test)

            if run_test(test, args, mocker):
                validate(test)
        finally:
            delete_file_path(test.get("export-path"))


def test_default_handlers(default_handler_tests, mocker):
    for test in default_handler_tests:
        cmd = build_namespace(test)
        try:
            delete_file_path(test.get("export-path"))
            create_file_path(test.get("export-path"))

            if run_test(test, cmd, mocker):
                validate(test)
        finally:
            delete_file_path(test.get("export-path"))


def test_zip_store(tmp_path, integ_tests, mocker):
    integ_tests = list(
        filter(lambda t: t["name"] == "packaging_zip_store_mar", integ_tests)
    )
    assert len(integ_tests) == 1
    test = integ_tests[0]

    test["export-path"] = tmp_path
    test["iterations"] = 1

    test["model-name"] = "zip-store"
    run_test(test, build_namespace(test), mocker)

    test["model-name"] = "zip"
    test["archive-format"] = "default"
    run_test(test, build_namespace(test), mocker)

    stored_size = Path(tmp_path).joinpath("zip-store.mar").stat().st_size
    zipped_size = Path(tmp_path).joinpath("zip.mar").stat().st_size

    assert zipped_size < stored_size


if __name__ == "__main__":
    test_model_archiver()
