import os
import sys
import tempfile
import urllib.request

from ts_scripts.shell_utils import rm_file

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(REPO_ROOT)

ROOT_DIR = f"{tempfile.gettempdir()}/workspace/"


def generate_densenet_test_model_archive():
    print("## Started densenet mar creation")
    model_store_dir = os.path.join(ROOT_DIR, "model_store")
    model_name = "densenet161_v1"
    version = "1.1"
    model_file = os.path.join(
        REPO_ROOT, "examples", "image_classifier", "densenet_161", "model.py"
    )
    serialized_model_file_name = "densenet161-8d451a50.pth"
    serialized_model_file_url = (
        f"https://download.pytorch.org/models/{serialized_model_file_name}"
    )
    serialized_file_path = os.path.join(model_store_dir, serialized_model_file_name)
    extra_files = os.path.join(
        REPO_ROOT, "examples", "image_classifier", "index_to_name.json"
    )
    handler = "image_classifier"

    os.makedirs(model_store_dir, exist_ok=True)
    os.chdir(model_store_dir)
    # Download & create DenseNet Model Archive
    urllib.request.urlretrieve(serialized_model_file_url, serialized_model_file_name)

    # create mar command
    cmd = f"torch-model-archiver \
                --model-name {model_name} \
                --version {version} \
                --model-file {model_file} \
                --serialized-file {serialized_file_path} \
                --extra-files {extra_files} \
                --handler {handler} \
                --force"
    print(f"## In directory: {os.getcwd()} | Executing command: {cmd}")
    sys_exit_code = os.system(cmd)
    os.remove(serialized_file_path)
    os.chdir(REPO_ROOT)
    return sys_exit_code


def run_pytest():
    print("## Started regression pytests")
    os.chdir(os.path.join(REPO_ROOT, "test", "pytest"))
    cmd = (
        "python -m grpc_tools.protoc -I ../../third_party/google/rpc --proto_path=../../frontend/server/src/main/resources/proto/"
        " --python_out=. --grpc_python_out=. ../../frontend/server/src/main/resources/proto/inference.proto"
        " ../../frontend/server/src/main/resources/proto/management.proto"
    )
    status = os.system(cmd)
    if status != 0:
        print("Could not generate gRPC client stubs")
        sys.exit(1)

    cmd = "python -m pytest -v ./ --ignore=sanity"
    print(f"## In directory: {os.getcwd()} | Executing command: {cmd}")
    status = os.system(cmd)
    rm_file("*_pb2*.py", True)
    return status


def test_regression():
    print("## Started regression tests")
    # generate_densenet_test_model_archive $MODEL_STORE
    model_creation_exit_code = generate_densenet_test_model_archive()
    py_test_exit_code = run_pytest()

    # If any one of the steps fail, exit with error
    if model_creation_exit_code != 0:
        sys.exit("## Densenet mar creation failed !")
    if py_test_exit_code != 0:
        sys.exit("## TorchServe Regression Pytests Failed")
