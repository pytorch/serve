import errno
import json
import os
import shutil
import subprocess

DEFAULT_MODEL_FILE = "model_archiver/tests/integ_tests/resources/regular_model/test_model.py"
DEFAULT_SERIALIZED_FILE = "model_archiver/tests/integ_tests/resources/regular_model/test_serialized_file.pt"
DEFAULT_HANDLER = "model_archiver/tests/integ_tests/resources/regular_model/test_handler.py"
DEFAULT_EXTRA_FILES = "model_archiver/tests/integ_tests/resources/regular_model/test_index_to_name.json"
DEFAULT_RUNTIME = "python"
DEFAULT_MODEL_NAME = "model"
DEFAULT_EXPORT_PATH = "/tmp/model"
MANIFEST_FILE = "MAR-INF/MANIFEST.json"


def update_tests(test):
    test["modelName"] = test.get("modelName", DEFAULT_MODEL_NAME)
    test["modelFile"] = test.get("modelPath", DEFAULT_MODEL_FILE)
    test["modelSerializedFile"] = test.get("modelSerializedFile", DEFAULT_SERIALIZED_FILE)
    test["handler"] = test.get("handler", DEFAULT_HANDLER)
    test["extraFiles"] = test.get("extraFiles", DEFAULT_EXTRA_FILES)
    test["runtime"] = test.get("runtime", DEFAULT_RUNTIME)
    test["exportPath"] = test.get("exportPath", DEFAULT_EXPORT_PATH)
    test["archiveFormat"] = test.get("archiveFormat", "default")
    return test


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


def run_test(test, cmd):
    it = test.get("iterations") if test.get("iterations") is not None else 1
    for i in range(it):
        try:
            subprocess.check_call(cmd, shell=True)
        except subprocess.CalledProcessError as exc:
            if test.get("expectError") is not True:
                assert 0, "{}".format(exc.output)
            else:
                return 0
    return 1


def validate_archive_exists(test):
    fmt = test.get("archiveFormat")
    if fmt == "tgz":
        assert os.path.isfile(os.path.join(test.get("exportPath"), test.get("modelName")+".tar.gz"))
    elif fmt == "no-archive":
        assert os.path.isdir(os.path.join(test.get("exportPath"), test.get("modelName")))
    else:
        assert os.path.isfile(os.path.join(test.get("exportPath"), test.get("modelName")+".mar"))


def validate_manifest_file(manifest, test):
    """
    Validate the MANIFEST file
    :param manifest:
    :param test:
    :return:
    """
    assert manifest.get("runtime") == test.get("runtime")
    assert manifest.get("model").get("modelName") == test.get("modelName")
    assert manifest.get("model").get("handler") == test.get("handler")


def validate_files(file_list, prefix):
    assert os.path.join(prefix, MANIFEST_FILE) in file_list
    assert os.path.join(prefix, "handler.py") in file_list
    assert os.path.join(prefix, "model.py") in file_list
    assert os.path.join(prefix, "model.pt") in file_list
    assert os.path.join(prefix, "dummy-artifacts.txt") in file_list
    assert os.path.join(prefix, "1.py") in file_list


def validate_tar_archive(test_cfg):
    import tarfile
    file_name = os.path.join(test_cfg.get("exportPath"), test_cfg.get("modelName") + ".tar.gz")
    f = tarfile.open(file_name, "r:gz")
    manifest = json.loads(f.extractfile(os.path.join(test_cfg.get("modelName"), MANIFEST_FILE)).read())
    validate_manifest_file(manifest, test_cfg)
    validate_files(f.getnames(), test_cfg.get("modelName"))


def validate_noarchive_archive(test):
    file_name = os.path.join(test.get("exportPath"), test.get("modelName"), MANIFEST_FILE)
    manifest = json.loads(open(file_name).read())
    validate_manifest_file(manifest, test)


def validate_mar_archive(test):
    import zipfile
    file_name = os.path.join(test.get("exportPath"), test.get("modelName") + ".mar")
    zf = zipfile.ZipFile(file_name, "r")
    manifest = json.loads(zf.open(MANIFEST_FILE).read())
    validate_manifest_file(manifest, test)


def validate_archive_content(test):
    fmt = test.get("archiveFormat")
    if fmt == "tgz":
        validate_tar_archive(test)
    if fmt == "no-archive":
        validate_noarchive_archive(test)
    if fmt == "default":
        validate_mar_archive(test)


def validate(test):
    validate_archive_exists(test)
    validate_archive_content(test)


def test_model_archiver():
    
    f = open("model_archiver/tests/integ_tests/configuration.json", "r")
    tests = json.loads(f.read())
    for t in tests:
        try:
            delete_file_path(t.get("exportPath"))
            create_file_path(t.get("exportPath"))
            t = update_tests(t)
            cmd = "torch-model-archiver " \
                  "--model-name {} " \
                  "--model-file {} " \
                  "--serialized-file {} " \
                  "--handler {} " \
                  "--extra-files {} " \
                  "--runtime {} " \
                  "--export-path {} " \
                  "--archive-format {}".format(t.get("modelName"),
                                               t.get("modelFile"),
                                               t.get("modelSerializedFile"),
                                               t.get("handler"),
                                               t.get("extraFiles"),
                                               t.get("runtime"),
                                               t.get("exportPath"),
                                               t.get("archiveFormat"))
            if t.get("force"):
                cmd += " -f"

            # TODO: Add tests to check for "convert" functionality
            if run_test(t, cmd):
                validate(t)
        finally:
            delete_file_path(t.get("exportPath"))


if __name__ == "__main__":
    test_model_archiver()
