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

def start_torchserve(ncs=False, model_store="model_store", models="", config_file="", log_file=""):
    print("Starting TorchServe")
    cmd = f"{torchserve_command[platform.system()]} --start --model-store={model_store}"
    if models:
        cmd += f" --models={models}"
    if ncs:
        cmd += " --ncs"
    if config_file:
        cmd += f" --ts-config={config_file}"
    if log_file:
        cmd += f" >> {log_file}"
    # cmd += " &"
    status = os.system(cmd)
    if status == 0:
        print("Successfully started TorchServe")
    else:
        print("TorchServe failed to start!")
        sys.exit(1)
    time.sleep(10)


def stop_torchserve():
    cmd = f"{torchserve_command[platform.system()]} --stop"
    os.system(cmd)
    time.sleep(10)


# Takes model name and mar name from model zoo as input
def register_model(model_name):
    print(f"Registering {model_name} model")
    response = None
    try:
        params = (
            ('model_name', model_name),
            ('url', f'https://torchserve.s3.amazonaws.com/mar_files/{model_name}.mar'),
            ('initial_workers', '1'),
            ('synchronous', 'true'),
        )
        response = requests.post("http://localhost:8081/models", params=params, verify=False)
    finally:
        if response and response.status_code == 200:
            print(f"Successfully registered {model_name} model with torchserve")
        else:
            print("Failed to register model with torchserve")
            sys.exit(1)


# Takes model URL and payload path as input
def run_inference(model_name, file_name):
    url = f"http://localhost:8080/predictions/{model_name}"
    print(f"Running inference on {model_name} model")
    for i in range(4):
        # Run inference
        response = None
        try:
            files = {
                'data': (file_name, open(file_name, 'rb')),
            }
            response = requests.post(url=url, files=files, timeout=120)
        finally:
            if response and response.status_code == 200:
                print(f"Successfully ran inference on {model_name} model.")
            else:
                print(f"Failed to run inference on {model_name} model")
                sys.exit(1)


def unregister_model(model_name):
    print(f"Unregistering {model_name} model")
    response = None
    try:
        response = requests.delete(f'http://localhost:8081/models/{model_name}', verify=False)
    finally:
        if response and response.status_code == 200:
            print(f"Successfully unregistered {model_name}")
        else:
            print(f"Failed to unregister {model_name}")
            sys.exit(1)
