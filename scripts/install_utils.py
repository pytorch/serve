import os
import platform
import time
import requests


torchserve_command = {
    "Windows": "torchserve.exe",
    "Darwin": "torchserve",
    "Linux": "torchserve"
}

def clean_slate():
    print("Cleaning up state")
    os.system('pip uninstall --yes torchserve || true')
    os.system('pip uninstall --yes torch-model-archiver || true')
    time.sleep(5)


def install_torch_deps_linux(is_gpu_instance, cuda_version):
    os.system('set +u')
    install_torch_deps(is_gpu_instance, cuda_version)
    os.system('set -u')


def install_torch_deps(is_gpu_instance, cuda_version):
    if is_gpu_instance and cuda_version == "cuda101":
        os.system('pip install -U -r requirements_gpu.txt')
    else:
        requirements = "requirements.txt"
        stable_build = ""
        if platform.system() == "Windows":
            requirements = "requirements_win.txt"
            stable_build = "-f https://download.pytorch.org/whl/torch_stable.html"
        os.system(f'pip install -U -r {requirements} {stable_build}')


def build_install_server():
    os.system('pip install .')


def build_install_archiver():
    execute_command('cd model-archiver;pip install .;cd ..', "Successfully installed torch-model-archiver",
                    "torch-model-archiver installation failed")

def clean_up_build_residuals():
    try:
        import shutil

        for (dirpath, dirnames, filenames) in os.walk(os.getcwd()):
            if '__pycache__' in dirnames:
                cache_dir = os.path.join(dirpath, '__pycache__')
                print(f'Cleaning - {cache_dir}')
                shutil.rmtree(cache_dir)
    except Exception as e:
        print('Error while cleaning cache file. Details - '+str(e))


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
        cmd += f" > {log_file}"
    cmd += " &"
    status = os.system(cmd)
    if status == 0:
        print("Successfully started TorchServe")
    else:
        print("TorchServe failed to start!")
        exit(1)
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
          exit(1)

# Takes model URL and payload path as input
def run_inference(model_name, file_name):
    url = f"http://localhost:8080/predictions/{model_name}"
    for i in range(4):
        print(f"Running inference on {model_name} model")
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
                exit(1)


def unregister_model(model_name):
    print("Unregistering $1 model")
    response = None

    try:
        response=response = requests.delete(f'http://localhost:8081/models/{model_name}', verify=False)
    except:
        if response.status_code == 200:
          print(f"Failed to register {model_name} model with torchserve")
          exit(1)
        else:
          print(f"Successfully registered {model_name} model with torchserve")


def install_pytest_suite_deps():
    os.system('pip install -U -r requirements/developer.txt')


def install_bert_dependencies():
    os.system('pip install transformers')


def run_markdown_link_checker():
  status = 0
  for (dirpath, dirnames, filenames) in os.walk(os.getcwd()):
    for file in filenames:
        if file.endswith('.md'):
            status = os.system(f'markdown-link-check {os.path.join(dirpath, file) } --config link_check_config.json')
            if status !=  0:
              print(f'Broken links in {os.path.join(dirpath, file) }')
              status = 1
  exit(status)


def execute_command(commands, success_msg, error_msg):
    if platform.system() == "Windows":
        commands = commands.replace(';','&')
    status = os.system(commands)
    if status == 0:
        print(success_msg)
    else:
        assert 0, error_msg
