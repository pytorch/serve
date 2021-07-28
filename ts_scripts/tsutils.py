import os
import platform
import sys
import time
import requests
from ts_scripts import marsgen as mg

torchserve_command = {
    "Windows": "torchserve.exe",
    "Darwin": "torchserve",
    "Linux": "torchserve"
}

torch_model_archiver_command = {
        "Windows": "torch-model-archiver.exe",
        "Darwin": "torch-model-archiver",
        "Linux": "torch-model-archiver"
    }

torch_workflow_archiver_command = {
        "Windows": "torch-workflow-archiver.exe",
        "Darwin": "torch-workflow-archiver",
        "Linux": "torch-workflow-archiver"
    }


def start_torchserve(
        ncs=False, model_store="model_store", workflow_store="",
        models="", config_file="", log_file="", wait_for=10, gen_mar=True):
    if gen_mar:
        mg.gen_mar(model_store)
    print("## Starting TorchServe")
    cmd = f"{torchserve_command[platform.system()]} --start --model-store={model_store}"
    if models:
        cmd += f" --models={models}"
    if workflow_store:
        cmd += f" --workflow-store={workflow_store}"
    if ncs:
        cmd += " --ncs"
    if config_file:
        cmd += f" --ts-config={config_file}"
    if log_file:
        print(f"## Console logs redirected to file: {log_file}")
        cmd += f" >> {log_file}"
    print(f"## In directory: {os.getcwd()} | Executing command: {cmd}")
    status = os.system(cmd)
    if status == 0:
        print("## Successfully started TorchServe")
        time.sleep(wait_for)
        return True
    else:
        print("## TorchServe failed to start !")
        return False


def stop_torchserve(wait_for=10):
    print("## Stopping TorchServe")
    cmd = f"{torchserve_command[platform.system()]} --stop"
    print(f"## In directory: {os.getcwd()} | Executing command: {cmd}")
    status = os.system(cmd)
    if status == 0:
        print("## Successfully stopped TorchServe")
        time.sleep(wait_for)
        return True
    else:
        print("## TorchServe failed to stop !")
        return False


# Takes model name and mar name from model zoo as input
def register_model(model_name, protocol="http", host="localhost", port="8081"):
    print(f"## Registering {model_name} model")
    model_zoo_url = "https://torchserve.s3.amazonaws.com"
    marfile = f"{model_name}.mar"
    if marfile not in mg.mar_set:
        marfile = f"{model_zoo_url}/mar_files/{model_name}.mar"

    params = (
        ("model_name", model_name),
        ("url", marfile),
        ("initial_workers", "1"),
        ("synchronous", "true"),
    )
    url = f"{protocol}://{host}:{port}/models"
    response = requests.post(url, params=params, verify=False)
    return response


def run_inference(model_name, file_name, protocol="http", host="localhost", port="8080", timeout=120):
    print(f"## Running inference on {model_name} model")
    url = f"{protocol}://{host}:{port}/predictions/{model_name}"
    files = {"data": (file_name, open(file_name, "rb"))}
    response = requests.post(url, files=files, timeout=timeout)
    return response


def unregister_model(model_name, protocol="http", host="localhost", port="8081"):
    print(f"## Unregistering {model_name} model")
    url = f"{protocol}://{host}:{port}/models/{model_name}"
    response = requests.delete(url, verify=False)
    return response


def generate_grpc_client_stubs():
    print("## Started generating gRPC clinet stubs")
    cmd = "python -m grpc_tools.protoc --proto_path=frontend/server/src/main/resources/proto/ --python_out=ts_scripts " \
          "--grpc_python_out=ts_scripts frontend/server/src/main/resources/proto/inference.proto " \
          "frontend/server/src/main/resources/proto/management.proto"
    status = os.system(cmd)
    if status != 0:
        print("Could not generate gRPC client stubs")
        sys.exit(1)


def register_workflow(workflow_name, protocol="http", host="localhost", port="8081"):
    print(f"## Registering {workflow_name} workflow")
    model_zoo_url = "https://torchserve.s3.amazonaws.com"
    params = (
        ("url", f"{model_zoo_url}/war_files/{workflow_name}.war"),
    )
    url = f"{protocol}://{host}:{port}/workflows"
    response = requests.post(url, params=params, verify=False)
    return response


def unregister_workflow(workflow_name, protocol="http", host="localhost", port="8081"):
    print(f"## Unregistering {workflow_name} workflow")
    url = f"{protocol}://{host}:{port}/workflows/{workflow_name}"
    response = requests.delete(url, verify=False)
    return response


def workflow_prediction(workflow_name, file_name, protocol="http", host="localhost", port="8080", timeout=120):
    print(f"## Running inference on {workflow_name} workflow")
    url = f"{protocol}://{host}:{port}/wfpredict/{workflow_name}"
    files = {"data": (file_name, open(file_name, "rb"))}
    response = requests.post(url, files=files, timeout=timeout)
    return response
