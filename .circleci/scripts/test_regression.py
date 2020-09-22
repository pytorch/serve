import os
import sys
import urllib.request

BASE_DIR = os.getcwd()

def generate_densenet_test_model_archive():
    model_store_dir = os.path.join("/", "workspace", "model_store")
    model_name = "densenet161_v1"
    version = "1.1"
    model_file = os.path.join(BASE_DIR, "examples", "image_classifier", "densenet_161", "model.py")
    serialized_model_file_name = "densenet161-8d451a50.pth"
    serialized_model_file_url = f"https://download.pytorch.org/models/{serialized_model_file_name}"
    serialized_file_path = os.path.join(model_store_dir, serialized_model_file_name)
    extra_files = os.path.join(BASE_DIR, "examples", "image_classifier", "index_to_name.json")
    handler = "image_classifier"

    os.makedirs(model_store_dir, exist_ok=True)
    os.chdir(model_store_dir)
    # Download & create DenseNet Model Archive
    urllib.request.urlretrieve(serialized_model_file_url, serialized_model_file_name)

    # create mar command
    sys_exit_code = os.system(f"torch-model-archiver \
                                    --model-name {model_name} \
                                    --version {version} \
                                    --model-file {model_file} \
                                    --serialized-file {serialized_file_path} \
                                    --extra-files {extra_files} \
                                    --handler {handler}")
    os.remove(serialized_file_path)
    return sys_exit_code


def run_pytest():
    os.chdir(os.path.join(BASE_DIR, "test", "pytest"))
    return os.system(f"python -m pytest -v ./")


# generate_densenet_test_model_archive $MODEL_STORE
MODEL_CREATION_EXIT_CODE = generate_densenet_test_model_archive()
PY_TEST_EXIT_CODE = run_pytest()

# If any one of the steps fail, exit with error
if any( EXIT_CODE != 0 for EXIT_CODE in [MODEL_CREATION_EXIT_CODE, PY_TEST_EXIT_CODE]):
    sys.exit("TorchServe Regression Tests Failed")