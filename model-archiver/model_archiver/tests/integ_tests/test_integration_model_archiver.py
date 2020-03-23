import errno
import json
import os
import shutil
import subprocess

DEFAULT_RUNTIME = "python"
MANIFEST_FILE = "MAR-INF/MANIFEST.json"


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
            if test.get("expect-error") is not True:
                assert 0, "{}".format(exc.output)
            else:
                return 0
    return 1


def validate_archive_exists(test):
    fmt = test.get("archive-format")
    if fmt == "tgz":
        assert os.path.isfile(os.path.join(test.get("export-path"), test.get("model-name")+".tar.gz"))
    elif fmt == "no-archive":
        assert os.path.isdir(os.path.join(test.get("export-path"), test.get("model-name")))
    else:
        assert os.path.isfile(os.path.join(test.get("export-path"), test.get("model-name")+".mar"))


def validate_manifest_file(manifest, test, default_handler=None):
    """
    Validate the MANIFEST file
    :param manifest:
    :param test:
    :return:
    """
    assert manifest.get("runtime") == test.get("runtime")
    assert manifest.get("model").get("modelName") == test.get("model-name")
    if not default_handler:
        assert manifest.get("model").get("handler") == test.get("handler").split("/")[-1]
    else:
        assert manifest.get("model").get("handler") == test.get("handler")


def validate_files(file_list, prefix, default_handler=None):
    assert os.path.join(prefix, MANIFEST_FILE) in file_list
    assert os.path.join(prefix, "test_handler.py") in file_list
    assert os.path.join(prefix, "test_model.py") in file_list
    assert os.path.join(prefix, "test_serialized_file.pt") in file_list
    assert os.path.join(prefix, "dummy-artifacts.txt") in file_list
    assert os.path.join(prefix, "1.py") in file_list

    if default_handler =="text_classifier":
        assert os.path.join(prefix, "source_vocab.pt") in file_list


def validate_tar_archive(test_cfg):
    import tarfile
    file_name = os.path.join(test_cfg.get("export-path"), test_cfg.get("model-name") + ".tar.gz")
    f = tarfile.open(file_name, "r:gz")
    manifest = json.loads(f.extractfile(os.path.join(test_cfg.get("model-name"), MANIFEST_FILE)).read())
    validate_manifest_file(manifest, test_cfg)
    validate_files(f.getnames(), test_cfg.get("model-name"))


def validate_noarchive_archive(test):
    file_name = os.path.join(test.get("export-path"), test.get("model-name"), MANIFEST_FILE)
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


def build_cmd(test):
    args = ['model-name', 'model-file', 'serialized-file', 'handler', 'extra-files', 'archive-format', 'source-vocab',
            'version', 'export-path', 'runtime']
    cmd = ["torch-model-archiver"]

    for arg in args:
        if arg in test:
            cmd.append("--{0} {1}".format(arg, test[arg]))

    return " ".join(cmd)


def test_model_archiver():
    with open("model_archiver/tests/integ_tests/configuration.json", "r") as f:
        tests = json.loads(f.read())
        for test in tests:
            try:
                delete_file_path(test.get("export-path"))
                create_file_path(test.get("export-path"))
                test["runtime"] = test.get("runtime", DEFAULT_RUNTIME)
                cmd = build_cmd(test)
                if test.get("force"):
                    cmd += " -f"

                if run_test(test, cmd):
                    validate(test)
            finally:
                delete_file_path(test.get("export-path"))


def test_default_handlers():
    with open("model_archiver/tests/integ_tests/default_handler_configuration.json", "r") as f:
        tests = json.loads(f.read())
        for test in tests:
            cmd = build_cmd(test)
            try:
                delete_file_path(test.get("export-path"))
                create_file_path(test.get("export-path"))

                if test.get("force"):
                    cmd += " -f"

                if run_test(test, cmd):
                    validate(test)
            finally:
                delete_file_path(test.get("export-path"))


if __name__ == "__main__":
    test_model_archiver()
