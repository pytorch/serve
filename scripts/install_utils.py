import os
import platform
import time
import subprocess

build_frontend_command = {"Windows": ".\\frontend\\gradlew.bat -p frontend clean build",
                          "Darwin": "frontend/gradlew -p frontend clean build",
                          "Linux": "frontend/gradlew -p frontend clean build"}


torchserve_command = {"Windows": "torchserve.exe",
                      "Darwin": "torchserve",
                      "Linux": "torchserve"}


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
        os.system('pip install -U -r requirements_gpu.txt -f https://download.pytorch.org/whl/torch_stable.html')
    else:
        os.system('pip install -U -r requirements.txt')


def install_pytest_suite_deps():
    os.system('pip install -U -r requirements/developer.txt')


def install_bert_dependencies():
  os.system('pip install transformers')


def build_frontend():
    execute_command(build_frontend_command[platform.system()], "Frontend build suite execution successful", "Frontend build suite execution failed!!! Check logs for more details")


def run_backend_pytest():
    execute_command('python -m pytest --cov-report html:htmlcov --cov=ts/ ts/tests/unit_tests/',
                    "Backend test suite execution successful", "Backend test suite execution failed!!! Check logs for more details")


def run_backend_python_linting():
    execute_command('pylint -rn --rcfile=./ts/tests/pylintrc ts/.', "Backend python linting suite execution successful"
                    "Backend python linting execution failed!!! Check logs for more details")


def run_model_archiver_python_linting():
  execute_command('cd model-archiver ; pylint -rn --rcfile=./model_archiver/tests/pylintrc model_archiver/. ; cd ..',
                  "Model archiver python linting suite execution successful", "Model archiver python linting execution failed!!! Check logs for more details")


def run_model_archiver_ut_suite():
    execute_command('cd model-archiver;python -m pytest --cov-report html:htmlcov_ut --cov=model_archiver/ model_archiver/tests/unit_tests/;cd ..',
                    "Model-archiver UT test suite execution successfully", "Model-archiver UT test suite execution failed!!! Check logs for more details")


def run_model_archiver_it_suite():
    execute_command('cd model-archiver;python -m pytest --cov-report html:htmlcov_it --cov=model_archiver/ model_archiver/tests/integ_tests/;cd ..',
                  "Model-archiver IT test suite execution successful", "Model-archiver IT test suite execution failed!!! Check logs for more details")


def build_install_server():
    os.system('pip install .')


def build_install_archiver():
    print(os.getcwd())
    execute_command('cd model-archiver;pip install .;cd ..', "Successfully installed torch-model-archiver",
                    "torch-model-archiver installation failed")


def start_torchserve():
    print("Starting TorchServe")
    proc = subprocess.Popen([torchserve_command[platform.system()], '--start --model-store model_store &'])
    if proc.pid:
        print("Successfully started TorchServe")
    else:
        print("TorchServe failed to start!")
        exit(1)

    time.sleep(10)


def stop_torchserve():
    os.system(torchserve_command[platform.system()]+' --stop')
    time.sleep(10)

clean_command = {"Windows": "del",
                      "Darwin": "rm -rf",
                      "Linux": "rm -rf"}

def clean_up_build_residuals():
    os.system(clean_command[platform.system()]+' ts/__pycache__/')
    os.system(clean_command[platform.system()]+' ts/metrics/__pycache__/')
    os.system(clean_command[platform.system()]+' ts/protocol/__pycache__/')
    os.system(clean_command[platform.system()]+' ts/utils/__pycache__/')


def execute_command(command, success_msg, error_msg):
    status = os.system(command)
    if status == 0:
        print(success_msg)
    else:
        assert 0, error_msg


"""# Takes model name and mar name from model zoo as input
register_model(model)
{
  echo "Registering $1 model"
  response=$(curl --write-out %{http_code} --silent --output /dev/null --retry 5 -X POST "http://localhost:8081/models?url=https://torchserve.s3.amazonaws.com/mar_files/{}.mar&initial_workers=1&synchronous=true&model_name={}".format(model, model))

  if [ ! "$response" == 200 ]
  then
      echo "Failed to register model with torchserve"
      cleanup
      exit 1
  else
      echo "Successfully registered $1 model with torchserve"
  fi
}

# Takes model URL and payload path as input
run_inference()
{
  for i in {1..4}
  do
    echo "Running inference on $1 model"
    response=$(curl --write-out %{http_code} --silent --output /dev/null --retry 5 -X POST http://localhost:8080/predictions/$1 -T $2)

    if [ ! "$response" == 200 ]
    then
        echo "Failed to run inference on $1 model"
        cleanup
        exit 1
    else
        echo "Successfully ran infernece on $1 model."
    fi
  done
}


unregister_model()
{
  echo "Unregistering $1 model"
  response=$(curl --write-out %{http_code} --silent --output /dev/null --retry 5 -X DELETE "http://localhost:8081/models/$1")

  if [ ! "$response" == 200 ]
  then
      echo "Failed to register $1 model with torchserve"
      cleanup
      exit 1
  else
      echo "Successfully registered $1 model with torchserve"
  fi
}

is_gpu_instance()
{
  [ -x "$(command -v nvidia-smi)" ]
}"""