import os
import sys
import tempfile
import urllib.request
import subprocess

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
    cmd = [
        "torch-model-archiver",
        "--model-name", model_name,
        "--version", version,
        "--model-file", model_file,
        "--serialized-file", serialized_file_path,
        "--extra-files", extra_files,
        "--handler", handler,
        "--force"
    ]
    print(f"## In directory: {os.getcwd()} | Executing command: {cmd}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    os.remove(serialized_file_path)
    os.chdir(REPO_ROOT)
    return result.returncode


def run_pytest():
    print("## Started regression pytests")
    proto_dir = os.path.join(REPO_ROOT, "frontend", "server", "src", "main", "resources", "proto")
    third_party_dir = os.path.join(REPO_ROOT, "third_party", "google", "rpc")
    proto_files = [
        "inference.proto",
        "management.proto"
    ]
    cmd = [
        "python", "-m", "grpc_tools.protoc",
        "-I", third_party_dir,
        "--proto_path", proto_dir
    ]
    cmd += ["--python_out=.", "--grpc_python_out=."] + [os.path.join(proto_dir, proto_file) for proto_file in proto_files]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("Could not generate gRPC client stubs")
        sys.exit(1)

    pytest_cmd = ["python", "-m", "pytest", "-v", "./", "--ignore=sanity"]
    print(f"## In directory: {os.getcwd()} | Executing command: {cmd}")
    result = subprocess.run(pytest_cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Pytest failed with return code {result.returncode}")

    rm_file("*_pb2*.py", True)
    return result.returncode


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
