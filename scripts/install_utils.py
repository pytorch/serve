import os
import platform
import time

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
        os.system('pip install -U -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html')


def build_install_server():
    os.system('pip install .')


def build_install_archiver():
    print(os.getcwd())
    execute_command('cd model-archiver\npip install .\ncd ..', "Successfully installed torch-model-archiver",
                    "torch-model-archiver installation failed")


def start_torchserve():
    print("Starting TorchServe")
    os.mkdir('model_store')
    status = os.system(torchserve_command[platform.system()]+' --start --model-store model_store &')
    if status == 0:
        print("Successfully started TorchServe")
    else:
        print("TorchServe failed to start!")
        exit(1)

    time.sleep(10)


def stop_torchserve():
    os.system(torchserve_command[platform.system()]+' --stop')
    time.sleep(10)

def clean_up_build_residuals():
    try:
        import shutil
        pwd = os.getcwd()
        shutil.rmtree('{}/ts/__pycache__'.format(pwd))
        shutil.rmtree('{}/ts/metrics/__pycache__/'.format(pwd))
        shutil.rmtree('{}/ts/protocol/__pycache__/'.format(pwd))
        shutil.rmtree('{}/ts/utils/__pycache__/'.format(pwd))
    except Exception as e:
        print('Error while cleaning cache file. Details - '+str(e))


def execute_command(command, success_msg, error_msg):
    from subprocess import Popen, PIPE
    process = Popen("cmd.exe", shell=False, universal_newlines=True,
                    stdin=PIPE, stderr=PIPE)
    out, err = process.communicate(command)

    if not err:
        print(success_msg)
    else:
        assert 0, error_msg

'''WIP - for sanity suite

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
'''