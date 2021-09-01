from os import path
import subprocess
import time
import glob
import json
import os
import requests
import sys
import tempfile

# To help discover margen modules
REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../")
sys.path.append(REPO_ROOT)
from ts_scripts import marsgen as mg

ROOT_DIR = f"{tempfile.gettempdir()}/workspace/"
MODEL_STORE = path.join(ROOT_DIR, "model_store/")
CODEBUILD_WD = path.abspath(path.join(__file__, "../../.."))

def start_torchserve(model_store=None, snapshot_file=None, no_config_snapshots=False, gen_mar=True):
    stop_torchserve()
    crate_mar_file_table()
    cmd = ["torchserve", "--start"]
    model_store = model_store if model_store else MODEL_STORE
    if gen_mar:
        mg.gen_mar(model_store)
    cmd.extend(["--model-store", model_store])
    if snapshot_file:
        cmd.extend(["--ts-config", snapshot_file])
    if no_config_snapshots:
        cmd.extend(["--no-config-snapshots"])
    print(cmd)
    subprocess.run(cmd)
    time.sleep(10)


def stop_torchserve():
    subprocess.run(["torchserve", "--stop"])
    time.sleep(10)


def delete_all_snapshots():
    for f in glob.glob('logs/config/*'):
        os.remove(f)
    assert len(glob.glob('logs/config/*')) == 0


def delete_model_store(model_store=None):
    """Removes all model mar files from model store"""
    model_store = model_store if model_store else MODEL_STORE
    for f in glob.glob(model_store + "/*.mar"):
        os.remove(f)


def torchserve_cleanup():
    stop_torchserve()
    delete_model_store()
    delete_all_snapshots()


def register_model(model_name, url):
    params = (
        ('model_name', model_name),
        ('url', url),
        ('initial_workers', '1'),
        ('synchronous', 'true'),
    )
    return register_model_with_params(params)


def register_model_with_params(params):
    response = requests.post('http://localhost:8081/models', params=params)
    return response


def unregister_model(model_name):
    response = requests.delete('http://localhost:8081/models/{}'.format(model_name))
    return response


def delete_mar_file_from_model_store(model_store=None, model_mar=None):
    model_store = model_store if (model_store is not None) else f"{ROOT_DIR}/model_store/"
    if model_mar is not None:
        for f in glob.glob(path.join(model_store, model_mar + "*")):
            os.remove(f)

environment_json = "/../postman/environment.json"
mar_file_table = {}
def crate_mar_file_table():
    if not mar_file_table:
        with open(os.path.dirname(__file__) + environment_json, 'rb') as f:
            env = json.loads(f.read())
        for item in env['values']:
            if item['key'].startswith('mar_path_'):
                mar_file_table[item['key']] = item['value']
