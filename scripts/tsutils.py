import os
import sys
import platform
import time
import requests


torchserve_command = {
    "Windows": "torchserve.exe",
    "Darwin": "torchserve",
    "Linux": "torchserve"
}


def is_gpu_instance():
    return True if os.system("nvidia-smi") == 0 else False


def is_conda_env():
    return True if os.system("conda") == 0 else False


def start_torchserve(ncs=False, model_store="model_store", models="", config_file="", log_file="", wait_for=10):
    print("## Starting TorchServe")
    cmd = f"{torchserve_command[platform.system()]} --start --model-store={model_store}"
    if models:
        cmd += f" --models={models}"
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
def register_model(model_name):
    print(f"## Registering {model_name} model")
    params = (
        ("model_name", model_name),
        ("url", f"https://torchserve.s3.amazonaws.com/mar_files/{model_name}.mar"),
        ("initial_workers", "1"),
        ("synchronous", "true"),
    )
    response = requests.post("http://localhost:8081/models", params=params, verify=False)
    return response


# Takes model URL and payload path as input
def run_inference(model_name, file_name):
    print(f"## Running inference on {model_name} model")
    url = f"http://localhost:8080/predictions/{model_name}"
    files = {"data": (file_name, open(file_name, "rb"))}
    response = requests.post(url=url, files=files, timeout=120)
    return response


def unregister_model(model_name):
    print(f"## Unregistering {model_name} model")
    response = requests.delete(f"http://localhost:8081/models/{model_name}", verify=False)
    return response
