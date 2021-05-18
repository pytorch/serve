from datetime import datetime
import errno
import json
import os
import shutil
import subprocess
import workflow_archiver

MANIFEST_FILE = "WAR-INF/MANIFEST.json"


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
    assert os.path.isfile(os.path.join(test.get("export-path"), test.get("workflow-name")+".war"))


def validate_manifest_file(manifest, test):
    """
    Validate the MANIFEST file
    :param manifest:
    :param test:
    :return:
    """
    assert datetime.strptime(manifest.get("createdOn"), "%d/%m/%Y %H:%M:%S")
    assert manifest.get("workflow").get("workflowName") == test.get("workflow-name")
    assert manifest.get("workflow").get("specFile") == test.get("spec-file").split("/")[-1]
    assert manifest.get("workflow").get("handler") == test.get("handler").split("/")[-1]
    assert manifest.get("archiverVersion") == workflow_archiver.__version__


def validate_war_archive(test):
    import zipfile
    file_name = os.path.join(test.get("export-path"), test.get("workflow-name") + ".war")
    zf = zipfile.ZipFile(file_name, "r")
    manifest = json.loads(zf.open(MANIFEST_FILE).read())
    validate_manifest_file(manifest, test)


def validate(test):
    validate_archive_exists(test)
    validate_war_archive(test)


def build_cmd(test):
    args = ['workflow-name', 'spec-file', 'handler', 'export-path']
    cmd = ["torch-workflow-archiver"]

    for arg in args:
        if arg in test:
            cmd.append("--{0} {1}".format(arg, test[arg]))

    return " ".join(cmd)


def test_workflow_archiver():
    with open("workflow_archiver/tests/integ_tests/configuration.json", "r") as f:
        tests = json.loads(f.read())
        for test in tests:
            try:
                delete_file_path(test.get("export-path"))
                create_file_path(test.get("export-path"))
                cmd = build_cmd(test)
                if test.get("force"):
                    cmd += " -f"

                if run_test(test, cmd):
                    validate(test)
            finally:
                delete_file_path(test.get("export-path"))


if __name__ == "__main__":
    test_workflow_archiver()
