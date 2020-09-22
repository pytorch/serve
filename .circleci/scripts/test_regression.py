import os
import sys
import urllib.request

BASE_DIR = os.getcwd()
MODEL_STORE_DIR=os.path.join("/", "workspace", "model_store")

def generate_densenet_test_model_archive(model_store_dir):
    os.makedirs(model_store_dir, exist_ok=True)
    os.chdir(model_store_dir)

    # Download & create DenseNet Model Archive
    MODEL_FILE_NAME = "densenet161-8d451a50.pth"
    urllib.request.urlretrieve(f"https://download.pytorch.org/models/{MODEL_FILE_NAME}", {MODEL_FILE_NAME})

    # create mar command
    cmd = f"torch-model-archiver --model-name densenet161_v1 \
            --version 1.1 \
            --model-file {BASE_DIR}/examples/image_classifier/densenet_161/model.py \
            --serialized-file $1/{MODEL_FILE_NAME} \
            --extra-files {BASE_DIR}/examples/image_classifier/index_to_name.json \
            --handler image_classifier"
    os.system(cmd)

    os.remove(MODEL_FILE_NAME)


def run_pytest():
    os.chdir(os.path.join(BASE_DIR, "test", "pytest"))
    return os.system(f"python -m pytest -v ./")

# generate_densenet_test_model_archive $MODEL_STORE
generate_densenet_test_model_archive(MODEL_STORE_DIR)
PY_TEST_EXIT_CODE = run_pytest()

sys.exit(PY_TEST_EXIT_CODE)