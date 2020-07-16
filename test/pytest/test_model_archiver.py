import subprocess
import time
import os
import glob
import requests
import json
from os import path

ROOT_DIR = "/workspace/"
MODEL_STORE = ROOT_DIR + "model_store/"
CODEBUILD_WD = path.abspath(path.join(__file__, "../../.."))


def start_torchserve(model_store=None, snapshot_file=None, no_config_snapshots=False):
    stop_torchserve()
    cmd = ["torchserve", "--start"]
    model_store = model_store if (model_store != None) else MODEL_STORE
    cmd.extend(["--model-store", model_store])
    if (snapshot_file != None):
        cmd.extend(["--ts-config", snapshot_file])
    if (no_config_snapshots):
        cmd.extend(["--no-config-snapshots"])
    subprocess.run(cmd)
    time.sleep(10)


def stop_torchserve():
    subprocess.run(["torchserve", "--stop"])
    time.sleep(5)


def delete_all_snapshots():
    for f in glob.glob('logs/config/*'):
        os.remove(f)
    assert len(glob.glob('logs/config/*')) == 0


def delete_model_store(model_store=None):
    '''Removes all model mar files from model store'''
    model_store = model_store if (model_store != None) else "/workspace/model_store/"
    for f in glob.glob(model_store + "/*.mar"):
        os.remove(f)


def torchserve_cleanup():
    stop_torchserve()
    delete_model_store()
    delete_all_snapshots()


def test_cleanup():
    torchserve_cleanup()


def test_multiple_model_versions_registration():
    # Download resnet-18 model
    cmd = ["wget", "https://download.pytorch.org/models/resnet18-5c106cde.pth"]
    subprocess.run(cmd)
    cmd2 = ["mv", "resnet18-5c106cde.pth", MODEL_STORE]
    subprocess.run(cmd2)
    # Use model archiver to create version 1.0 of resnet-18
    v1_cmd = "torch-model-archiver --model-name resnet-18 --version 1.0 --model-file " \
             + CODEBUILD_WD + "/examples/image_classifier/resnet_18/model.py --serialized-file "\
             + MODEL_STORE + "/resnet18-5c106cde.pth --handler image_classifier --extra-files " \
             + CODEBUILD_WD + "/examples/image_classifier/index_to_name.json"
    v1_cmd_list = v1_cmd.split(" ")
    # Use model archiver to create version 2.0 of resnet-18
    v2_cmd = "torch-model-archiver --model-name resnet-18_v2 --version 2.0 --model-file " \
             + CODEBUILD_WD + "/examples/image_classifier/resnet_18/model.py --serialized-file " \
             + MODEL_STORE + "/resnet18-5c106cde.pth --handler image_classifier --extra-files " \
             + CODEBUILD_WD + "/examples/image_classifier/index_to_name.json"
    v2_cmd_list = v2_cmd.split(" ")
    subprocess.run(v1_cmd_list)
    subprocess.run(v2_cmd_list)
    cmd3 = ["mv", "resnet-18.mar", MODEL_STORE]
    cmd4 = ["mv", "resnet-18_v2.mar", MODEL_STORE]
    subprocess.run(cmd3)
    subprocess.run(cmd4)

    start_torchserve(no_config_snapshots=True)
    params1 = (
        ('model_name', 'resnet18'),
        ('url', 'resnet-18.mar'),
        ('initial_workers', '3'),
        ('synchronous', 'true'),
    )
    # Register model with model_name=resnet18 using local mar file for version 1.0
    response = requests.post('http://localhost:8081/models', params=params1)
    time.sleep(15)
    params2 = (
        ('model_name', 'resnet18'),
        ('url', 'resnet-18_v2.mar'),
        ('initial_workers', '3'),
        ('synchronous', 'true'),
    )
    # Register model with model_name=resnet18 using local mar file for version 2.0
    response = requests.post('http://localhost:8081/models', params=params2)
    time.sleep(15)
    response = requests.get('http://localhost:8081/models/resnet18/all')
    time.sleep(5)
    # Verify that we can use the list models api to get all versions of resnet-18
    assert len(json.loads(response.content)) == 2


def test_duplicate_model_registration_using_local_url_followed_by_http_url():
    # Registration through local mar url is already complete in previous test case.
    # Now try to register same model using http url in this next step
    response = requests.post(
        'http://127.0.0.1:8081/models?url=https://torchserve.s3.amazonaws.com/mar_files/resnet-18.mar')
    time.sleep(15)
    if json.loads(response.content)['code'] == 500 and \
            json.loads(response.content)['type'] == "InternalServerException":
        assert True, "Internal Server Exception, " \
                      "Model file already exists!! Duplicate model registration request"
        response = requests.delete('http://localhost:8081/models/resnet18')
        time.sleep(10)
    else:
        assert False, "Something is not right!! Successfully re-registered existing model "

        # torchserve_cleanup()


def test_duplicate_model_registration_using_http_url_followed_by_local_url():
    # delete_model_store()
    # start_torchserve(no_config_snapshots=True)
    #Register using http url
    requests.post(
        'http://127.0.0.1:8081/models?url=https://torchserve.s3.amazonaws.com/mar_files/resnet-18.mar')
    time.sleep(15)
    # Download resnet-18 model serialized file
    cmd = ["wget", "https://download.pytorch.org/models/resnet18-5c106cde.pth"]
    subprocess.run(cmd)
    cmd2 = ["mv", "resnet18-5c106cde.pth", MODEL_STORE]
    subprocess.run(cmd2)
    v1_cmd = "torch-model-archiver --model-name resnet-18 --version 1.0 --model-file " \
             + CODEBUILD_WD + "/examples/image_classifier/resnet_18/model.py --serialized-file "\
             + MODEL_STORE + "/resnet18-5c106cde.pth --handler image_classifier --extra-files " \
             + CODEBUILD_WD + "/examples/image_classifier/index_to_name.json"
    v1_cmd_list = v1_cmd.split(" ")
    # Use model archiver to create version 1.0 of resnet-18
    subprocess.run(v1_cmd_list)
    cmd3 = ["mv", "resnet-18.mar", MODEL_STORE]
    subprocess.run(cmd3)
    params1 = (
        ('model_name', 'resnet18'),
        ('url', 'resnet-18.mar'),
        ('initial_workers', '3'),
        ('synchronous', 'true'),
    )
    # Register using local mar url
    response = requests.post('http://localhost:8081/models', params=params1)
    time.sleep(15)

    if json.loads(response.content)['code'] == 409 and \
            json.loads(response.content)['type'] == "ConflictStatusException":
        assert True, "Conflict Status Exception, " \
                      "Duplicate model registration request"
        response = requests.delete('http://localhost:8081/models/resnet18')
        time.sleep(10)
    else:
        assert False, "Something is not right!! Successfully re-registered existing model "


def run_model_archiver_to_regenerate_model_mar(force_flag=None):
    delete_model_store()
    # Download resnet-18 model serialized file
    cmd = ["wget", "https://download.pytorch.org/models/resnet18-5c106cde.pth"]
    subprocess.run(cmd)
    cmd2 = ["mv", "resnet18-5c106cde.pth", MODEL_STORE]
    subprocess.run(cmd2)
    v1_cmd = "torch-model-archiver --model-name resnet-18 --version 1.0 --model-file " \
             + CODEBUILD_WD + "/examples/image_classifier/resnet_18/model.py --serialized-file "\
             + MODEL_STORE + "/resnet18-5c106cde.pth --handler image_classifier --extra-files " \
             + CODEBUILD_WD + "/examples/image_classifier/index_to_name.json"
    v1_cmd_list = v1_cmd.split(" ")
    subprocess.run(v1_cmd_list)
    # Now regenerate the same model file using same process as above without force flag
    v2_cmd = "torch-model-archiver --model-name resnet-18 --version 1.0 --model-file " \
             + CODEBUILD_WD + "/examples/image_classifier/resnet_18/model.py --serialized-file "\
             + MODEL_STORE + "/resnet18-5c106cde.pth --handler image_classifier --extra-files " \
             + CODEBUILD_WD + "/examples/image_classifier/index_to_name.json"
    v2_cmd_list = v2_cmd.split(" ")
    if force_flag != None:
        v2_cmd_list.extend([force_flag])
    return subprocess.run(v2_cmd_list).returncode


def test_model_archiver_to_regenerate_model_mar_without_force():
    response = run_model_archiver_to_regenerate_model_mar()
    try:
        assert (0 != response), "Mar file couldn't be created.use -f option"
    finally:
        for f in glob.glob("resnet*.mar"):
            os.remove(f)


def test_model_archiver_to_regenerate_model_mar_with_force():
    response = run_model_archiver_to_regenerate_model_mar("--force")
    try:
        assert (0 == response), "Successfully created Mar file by using -f option"
    finally:
        for f in glob.glob("resnet*.mar"):
            os.remove(f)


def test_model_archiver_without_handler_flag():
    delete_model_store()
    cmd = ["wget", "https://download.pytorch.org/models/resnet18-5c106cde.pth"]
    subprocess.run(cmd)
    cmd2 = ["mv", "resnet18-5c106cde.pth", MODEL_STORE]
    subprocess.run(cmd2)
    v1_cmd = "torch-model-archiver --model-name resnet-18 --version 1.0 --model-file " \
             + CODEBUILD_WD + "/examples/image_classifier/resnet_18/model.py --serialized-file "\
             + MODEL_STORE + "/resnet18-5c106cde.pth --extra-files " \
             + CODEBUILD_WD + "/examples/image_classifier/index_to_name.json"
    v1_cmd_list = v1_cmd.split(" ")
    try:
        assert (0 != subprocess.run(v1_cmd_list).returncode), "Mar file couldn't be created." \
                                                              "No handler specified"
    finally:
        for f in glob.glob("resnet*.mar"):
            os.remove(f)


def test_model_archiver_without_model_name_flag():
    delete_model_store()
    cmd = ["wget", "https://download.pytorch.org/models/resnet18-5c106cde.pth"]
    subprocess.run(cmd)
    cmd2 = ["mv", "resnet18-5c106cde.pth", MODEL_STORE]
    subprocess.run(cmd2)
    v1_cmd = "torch-model-archiver --version 1.0 --model-file " \
             + CODEBUILD_WD + "/examples/image_classifier/resnet_18/model.py --serialized-file "\
             + MODEL_STORE + "/resnet18-5c106cde.pth --handler image_classifier --extra-files " \
             + CODEBUILD_WD + "/examples/image_classifier/index_to_name.json"
    v1_cmd_list = v1_cmd.split(" ")
    try:
        assert (0 != subprocess.run(v1_cmd_list).returncode), "Mar file couldn't be created." \
                                                              "No model_name specified"
    finally:
        for f in glob.glob("resnet*.mar"):
            os.remove(f)


def test_model_archiver_without_model_file_flag():
    delete_model_store()
    cmd = ["wget", "https://download.pytorch.org/models/resnet18-5c106cde.pth"]
    subprocess.run(cmd)
    cmd2 = ["mv", "resnet18-5c106cde.pth", MODEL_STORE]
    subprocess.run(cmd2)
    v1_cmd = "torch-model-archiver --model-name resnet-18 --version 1.0 --serialized-file "\
             + MODEL_STORE + "/resnet18-5c106cde.pth --handler image_classifier --extra-files " \
             + CODEBUILD_WD + "/examples/image_classifier/index_to_name.json"
    v1_cmd_list = v1_cmd.split(" ")
    try:
        assert (0 == subprocess.run(v1_cmd_list).returncode)
    finally:
        for f in glob.glob("resnet*.mar"):
            os.remove(f)


def test_model_archiver_without_serialized_flag():
    delete_model_store()
    cmd = ["wget", "https://download.pytorch.org/models/resnet18-5c106cde.pth"]
    subprocess.run(cmd)
    cmd2 = ["mv", "resnet18-5c106cde.pth", MODEL_STORE]
    subprocess.run(cmd2)
    v1_cmd = "torch-model-archiver --model-name resnet-18 --version 1.0 --model-file " \
             + CODEBUILD_WD + "/examples/image_classifier/resnet_18/model.py" \
                              " --handler image_classifier --extra-files " \
             + CODEBUILD_WD + "/examples/image_classifier/index_to_name.json"
    v1_cmd_list = v1_cmd.split(" ")
    try:
        assert (0 != subprocess.run(v1_cmd_list).returncode), "Mar file couldn't be created." \
                                                              "No serialized flag specified"
    finally:
        for f in glob.glob("resnet*.mar"):
            os.remove(f)
            torchserve_cleanup()