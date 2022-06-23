import subprocess
import time
import os
import glob
import requests
import json
import test_utils

MODEL_SFILE_NAME = 'resnet18-f37072fd.pth'


def setup_module(module):
    test_utils.torchserve_cleanup()
    response = requests.get('https://download.pytorch.org/models/' + MODEL_SFILE_NAME, allow_redirects=True)
    with open(os.path.join(test_utils.MODEL_STORE, MODEL_SFILE_NAME), 'wb') as f:
        f.write(response.content)


def teardown_module(module):
    test_utils.torchserve_cleanup()


def create_resnet_archive(model_name="resnset-18", version="1.0", force=False):
    cmd = test_utils.model_archiver_command_builder(
        model_name,
        version,
        os.path.join(test_utils.CODEBUILD_WD, "examples", "image_classifier", "resnet_18", "model.py"),
        os.path.join(test_utils.MODEL_STORE, "resnet18-f37072fd.pth"),
        "image_classifier",
        os.path.join(test_utils.CODEBUILD_WD, "examples", "image_classifier", "index_to_name.json"),
        force
    )
    print(cmd)
    cmd = cmd.split(" ")

    return subprocess.run(cmd).returncode


def clean_mar_file(mar_name):
    path = os.path.join(test_utils.MODEL_STORE, mar_name)
    if os.path.exists(path):
        os.remove(path)


def test_multiple_model_versions_registration():
    # Download resnet-18 model

    create_resnet_archive("resnet-18", "1.0")
    create_resnet_archive("resnet-18_v2", "2.0")

    test_utils.start_torchserve(no_config_snapshots=True)

    response = requests.get('http://localhost:8081/models/resnet18/all')
    print(response.content)

    test_utils.register_model("resnet18", "resnet-18.mar")
    test_utils.register_model("resnet18", "resnet-18_v2.mar")

    response = requests.get('http://localhost:8081/models/resnet18/all')
    time.sleep(5)
    # Verify that we can use the list models api to get all versions of resnet-18
    assert len(json.loads(response.content)) == 2


def test_duplicate_model_registration_using_local_url_followed_by_http_url():
    # Registration through local mar url is already complete in previous test case.
    # Now try to register same model using http url in this next step
    response = test_utils.register_model("resnet18", "https://torchserve.pytorch.org/mar_files/resnet-18.mar")
    time.sleep(15)
    if json.loads(response.content)['code'] == 500 and \
            json.loads(response.content)['type'] == "InternalServerException":
        assert True, "Internal Server Exception, " \
                     "Model file already exists!! Duplicate model registration request"
        test_utils.unregister_model("resnet18")
        time.sleep(10)
    else:
        assert False, "Something is not right!! Successfully re-registered existing model "


def test_duplicate_model_registration_using_http_url_followed_by_local_url():
    # Register using http url
    clean_mar_file("resnet-18.mar")
    response = test_utils.register_model("resnet18", "https://torchserve.pytorch.org/mar_files/resnet-18.mar")

    create_resnet_archive()
    response = test_utils.register_model("resnet18", "resnet-18.mar")

    if json.loads(response.content)['code'] == 409 and \
            json.loads(response.content)['type'] == "ConflictStatusException":
        assert True, "Conflict Status Exception, " \
                     "Duplicate model registration request"
        response = test_utils.unregister_model("resnet18")
        time.sleep(10)
    else:
        assert False, "Something is not right!! Successfully re-registered existing model "


def test_model_archiver_to_regenerate_model_mar_without_force():
    clean_mar_file("resnet-18.mar")
    response = create_resnet_archive("resnet-18", "1.0")
    response = create_resnet_archive("resnet-18", "1.0")
    try:
        assert (0 != response), "Mar file couldn't be created.use -f option"
    finally:
        for f in glob.glob("resnet*.mar"):
            os.remove(f)


def test_model_archiver_to_regenerate_model_mar_with_force():
    clean_mar_file("resnet-18.mar")
    response = create_resnet_archive("resnet-18", "1.0")
    response = create_resnet_archive("resnet-18", "1.0", force=True)
    try:
        assert (0 == response), "Successfully created Mar file by using -f option"
    finally:
        for f in glob.glob("resnet*.mar"):
            os.remove(f)


def test_model_archiver_without_handler_flag():
    cmd = test_utils.model_archiver_command_builder(
        "resnet-18",
        "1.0",
        os.path.join(test_utils.CODEBUILD_WD, "examples", "image_classifier", "resnet_18", "model.py"),
        os.path.join(test_utils.MODEL_STORE, "resnet18-f37072fd.pth"),
        None,
        os.path.join(test_utils.CODEBUILD_WD, "examples", "image_classifier", "index_to_name.json")
    )
    cmd = cmd.split(" ")
    try:
        assert (0 != subprocess.run(cmd).returncode), "Mar file couldn't be created." \
                                                              "No handler specified"
    finally:
        for f in glob.glob("resnet*.mar"):
            os.remove(f)


def test_model_archiver_without_model_name_flag():
    cmd = test_utils.model_archiver_command_builder(
        None,
        "1.0",
        os.path.join(test_utils.CODEBUILD_WD, "examples", "image_classifier", "resnet_18", "model.py"),
        os.path.join(test_utils.MODEL_STORE, "resnet18-f37072fd.pth"),
        "image_classifier",
        os.path.join(test_utils.CODEBUILD_WD, "examples", "image_classifier", "index_to_name.json")
    )
    cmd = cmd.split(" ")
    assert (0 != subprocess.run(cmd).returncode), "Mar file couldn't be created." \
                                                          "No model_name specified"


def test_model_archiver_without_model_file_flag():
    cmd = test_utils.model_archiver_command_builder(
        "resnet-18",
        "1.0",
        None,
        os.path.join(test_utils.MODEL_STORE, "resnet18-f37072fd.pth"),
        "image_classifier",
        os.path.join(test_utils.CODEBUILD_WD, "examples", "image_classifier", "index_to_name.json"),
        True
    )

    cmd = cmd.split(" ")
    try:
        assert (0 == subprocess.run(cmd).returncode)
    finally:
        for f in glob.glob("resnet*.mar"):
            os.remove(f)


def test_model_archiver_without_serialized_flag():
    cmd = test_utils.model_archiver_command_builder(
        "resnet-18",
        "1.0",
        os.path.join(test_utils.CODEBUILD_WD, "examples", "image_classifier", "resnet_18", "model.py"),
        None,
        "image_classifier",
        os.path.join(test_utils.CODEBUILD_WD, "examples", "image_classifier", "index_to_name.json")
    )

    cmd = cmd.split(" ")
    assert (0 != subprocess.run(cmd).returncode), "Mar file couldn't be created." \
                                                          "No serialized flag specified"
