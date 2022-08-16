import os
import subprocess
import requests
import test_utils

from shutil import copy
from PIL import Image
from io import BytesIO

CURR_FILE_PATH = os.path.dirname(os.path.realpath(__file__))
REPO_ROOT_DIR = os.path.normpath(os.path.join(CURR_FILE_PATH, "..", ".."))
MODELSTORE_DIR = os.path.join(REPO_ROOT_DIR, "modelstore")
DCGAN_EXAMPLE_DIR = os.path.join(REPO_ROOT_DIR, "examples", "dcgan_fashiongen")
DCGAN_MAR_FILE = os.path.join(DCGAN_EXAMPLE_DIR, "dcgan_fashiongen.mar")


os.environ['MKL_THREADING_LAYER'] = 'GNU'
# Work around for issue - https://github.com/pytorch/pytorch/issues/37377

def setup_module():
    test_utils.torchserve_cleanup()
    create_example_mar()

    os.makedirs(MODELSTORE_DIR, exist_ok=True)  # Create modelstore directory
    copy(DCGAN_MAR_FILE, MODELSTORE_DIR)  # Copy *.mar to modelstore

    test_utils.start_torchserve(model_store=MODELSTORE_DIR)
    pass


def teardown_module():
    test_utils.torchserve_cleanup()

    # Empty and Remove modelstore directory
    test_utils.delete_model_store(MODELSTORE_DIR)
    os.rmdir(MODELSTORE_DIR)

    delete_example_mar()
    pass


def create_example_mar():
    # Create only if not already present
    if not os.path.exists(DCGAN_MAR_FILE):
        create_mar_cmd = "cd " + DCGAN_EXAMPLE_DIR + ";./create_mar.sh"
        subprocess.check_call(create_mar_cmd, shell=True)


def delete_example_mar():
    try:
        os.remove(DCGAN_MAR_FILE)
    except OSError:
        pass


def test_model_archive_creation():
    # *.mar created in setup phase
    assert os.path.exists(DCGAN_MAR_FILE), "Failed to create dcgan mar file"


def test_model_register_unregister():
    reg_resp = test_utils.register_model("dcgan_fashiongen", "dcgan_fashiongen.mar")
    assert reg_resp.status_code == 200, "Model Registration Failed"

    unreg_resp = test_utils.unregister_model("dcgan_fashiongen")
    assert unreg_resp.status_code == 200, "Model Unregistration Failed"


def test_image_generation_without_any_input_constraints():
    test_utils.register_model("dcgan_fashiongen", "dcgan_fashiongen.mar")
    input_json = {}
    response = requests.post(url="http://localhost:8080/predictions/dcgan_fashiongen", json=input_json)
    fp = BytesIO(response.content)
    img = Image.open(fp)
    # Expect a jpeg of dimension 64 x 64, it contains only 1 image by default
    assert response.status_code == 200, "Image generation failed"
    assert img.get_format_mimetype() == "image/jpeg", "Generated image is not a jpeg"
    assert img.size == (64, 64), "Generated image is not of correct dimensions"
    test_utils.unregister_model("dcgan_fashiongen")


def test_image_generation_with_input_constraints():
    test_utils.register_model("dcgan_fashiongen", "dcgan_fashiongen.mar")
    input_json = {
        "number_of_images": 64,
        "input_gender": "Men",
        "input_category": "SHIRTS",
        "input_pose": "id_gridfs_1"
    }
    response = requests.post(url="http://localhost:8080/predictions/dcgan_fashiongen", json=input_json)
    fp = BytesIO(response.content)
    img = Image.open(fp)
    # Expect a jpeg of dimension 530 x 530, it contains 64 images
    assert response.status_code == 200, "Image generation failed"
    assert img.get_format_mimetype() == "image/jpeg", "Generated image is not a jpeg"
    assert img.size == (530, 530), "Generated image is not of correct dimensions"
    test_utils.unregister_model("dcgan_fashiongen")
