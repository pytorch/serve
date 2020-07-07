import subprocess
import time
import os
import glob
import requests
import json

ROOT_DIR="/workspace/"
MODEL_STORE=ROOT_DIR+"model_store"
#CHANGE THIS TO CORRECT PYTORCH CODE REPOSITORY
CODEBUILD_WD="/home/deepak/projects/ofc/db_torch/torch_issue_394/serve"

def start_torchserve(model_store=None, snapshot_file=None, no_config_snapshots=False):
    stop_torchserve()
    cmd = ["torchserve", "--start"]
    model_store = model_store if (model_store != None) else "/workspace/model_store/"
    cmd.extend(["--model-store", "/workspace/model_store/"])
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
    model_store = model_store if (model_store != None) else "/workspace/model_store/"
    for f in glob.glob(model_store+"/*"):
        os.remove(f)


def torchserve_cleanup():
    stop_torchserve()
    delete_model_store()
    delete_all_snapshots()


def test_multiple_model_versions_registration():
    torchserve_cleanup()
    # Download resnet-18 model
    cmd = ["wget", "https://download.pytorch.org/models/resnet18-5c106cde.pth"]
    subprocess.run(cmd)
    cmd2 = ["mv", "resnet18-5c106cde.pth", MODEL_STORE]
    subprocess.run(cmd2)
    v1_cmd = "torch-model-archiver --model-name resnet-18 --version 1.0 --model-file " \
          +CODEBUILD_WD+"/examples/" \
          "image_classifier/resnet_18/model.py --serialized-file "+MODEL_STORE+"/resnet18-5c106cde.pth" \
          " --handler image_classifier --extra-files " \
          +CODEBUILD_WD+"/examples" \
          "/image_classifier/index_to_name.json"
    v1_cmd_list = v1_cmd.split(" ")
    v2_cmd = "torch-model-archiver --model-name resnet-18_v2 --version 2.0 --model-file " \
             +CODEBUILD_WD+"/examples/" \
          "image_classifier/resnet_18/model.py --serialized-file "+MODEL_STORE+"/resnet18-5c106cde.pth" \
          " --handler image_classifier --extra-files " \
          +CODEBUILD_WD+"/examples" \
             "/image_classifier/index_to_name.json"
    v2_cmd_list = v2_cmd.split(" ")
    subprocess.run(v1_cmd_list)
    subprocess.run(v2_cmd_list)
    cmd3 = ["mv", "resnet-18.mar", MODEL_STORE]
    cmd4 = ["mv", "resnet-18_v2.mar", MODEL_STORE]
    subprocess.run(cmd3)
    subprocess.run(cmd4)
    # Use model archiver to create version 1.0 of resnet-18
    # Use model archiver to create version 2.0 of resnet-18
    # Register model with model_name=resnet18 using local mar file for version 1.0
    # Register model with model_name=resnet18 using local mar file for version 2.0
    # Verify that we can use the list models api to get all versions of resnet-18
    start_torchserve(no_config_snapshots=True)
    params1 = (
        ('model_name', 'resnet18'),
        ('url', 'resnet-18.mar'),
        ('initial_workers', '3'),
        ('synchronous', 'true'),
    )

    response = requests.post('http://localhost:8081/models', params=params1)
    time.sleep(15)
    params2 = (
        ('model_name', 'resnet18'),
        ('url', 'resnet-18_v2.mar'),
        ('initial_workers', '3'),
        ('synchronous', 'true'),
    )

    response = requests.post('http://localhost:8081/models', params=params2)
    time.sleep(15)
    response = requests.get('http://localhost:8081/models/resnet18/all')
    print(json.loads(response.content))
    time.sleep(5)
    assert len(json.loads(response.content)) == 2
    torchserve_cleanup()


def test_duplicate_model_registration_using_local_and_http_url():
    start_torchserve()
    requests.post(
        'http://127.0.0.1:8081/models?url=https://torchserve.s3.amazonaws.com/mar_files/resnet-18.mar')
    time.sleep(15)
    # Download resnet-18 model serialized file
    cmd = ["wget", "https://download.pytorch.org/models/resnet18-5c106cde.pth"]
    subprocess.run(cmd)
    cmd2 = ["mv", "resnet18-5c106cde.pth", MODEL_STORE]
    subprocess.run(cmd2)
    v1_cmd = "torch-model-archiver --model-name resnet-18 --version 1.0 --model-file " \
          +CODEBUILD_WD+"/examples/" \
          "image_classifier/resnet_18/model.py --serialized-file "+MODEL_STORE+"/resnet18-5c106cde.pth" \
          " --handler image_classifier --extra-files " \
          +CODEBUILD_WD+"/examples" \
          "/image_classifier/index_to_name.json"
    v1_cmd_list = v1_cmd.split(" ")
    subprocess.run(v1_cmd_list)
    cmd3 = ["mv", "resnet-18.mar", MODEL_STORE]
    subprocess.run(cmd3)
    # Use model archiver to create version 1.0 of resnet-18
    # Use model archiver to create version 2.0 of resnet-18
    # Register model with model_name=resnet18 using local mar file for version 1.0
    # Register model with model_name=resnet18 using local mar file for version 2.0
    # Verify that we can use the list models api to get all versions of resnet-18
    params1 = (
        ('model_name', 'resnet-18'),
        ('url', 'resnet-18.mar'),
        ('initial_workers', '3'),
        ('synchronous', 'true'),
    )
    response = requests.post('http://localhost:8081/models', params=params1)
    time.sleep(15)
    try:
        if json.loads(response.content)['code'] == 409 and \
                json.loads(response.content)['type'] == "ConflictStatusException":
            assert False, "Conflict Status Exception, " \
                          "Duplicate model registration request"
        else:
            assert True, "Successfully registered model "
    finally:
        torchserve_cleanup()

def test_model_archiver_to_regenerate_model_mar_without_force_flag():
    torchserve_cleanup()
    # Download resnet-18 model serialized file
    cmd = ["wget", "https://download.pytorch.org/models/resnet18-5c106cde.pth"]
    subprocess.run(cmd)
    cmd2 = ["mv", "resnet18-5c106cde.pth", MODEL_STORE]
    subprocess.run(cmd2)
    v1_cmd = "torch-model-archiver --model-name resnet-18 --version 1.0 --model-file " \
          +CODEBUILD_WD+"/examples/" \
          "image_classifier/resnet_18/model.py --serialized-file "+MODEL_STORE+"/resnet18-5c106cde.pth" \
          " --handler image_classifier --extra-files " \
          +CODEBUILD_WD+"/examples" \
          "/image_classifier/index_to_name.json"
    v1_cmd_list = v1_cmd.split(" ")
    retval1 = subprocess.run(v1_cmd_list)
    #Now regenerate the same model file using same process as above without force flag
    v2_cmd = "torch-model-archiver --model-name resnet-18 --version 1.0 --model-file " \
          +CODEBUILD_WD+"/examples/" \
          "image_classifier/resnet_18/model.py --serialized-file "+MODEL_STORE+"/resnet18-5c106cde.pth" \
          " --handler image_classifier --extra-files " \
          +CODEBUILD_WD+"/examples" \
          "/image_classifier/index_to_name.json"
    v2_cmd_list = v2_cmd.split(" ")
    try:
        assert (0 == subprocess.run(v2_cmd_list).returncode), "Mar file couldn't be created.use -f option"
    finally:
        torchserve_cleanup()