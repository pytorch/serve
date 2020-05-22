import subprocess
import time
import os
import glob
import requests
import json

def start_torchserve(model_store=None, snapshot_file=None, no_config_snapshots=False):
    stop_torchserve()
    cmd = ["torchserve","--start"]
    model_store = model_store if (model_store != None) else "/workspace/model_store/"
    cmd.extend(["--model-store", "/workspace/model_store/"])
    if(snapshot_file != None):
        cmd.extend(["--ts-config", snapshot_file])
    if(no_config_snapshots):
        cmd.extend(["--no-config-snapshots"])
    subprocess.run(cmd)
    time.sleep(10)


def stop_torchserve():
    subprocess.run(["torchserve", "--stop"])
    time.sleep(5)


def delete_all_snapshots():
    for f in glob.glob('logs/config/*'):
        os.remove(f)
    assert len(glob.glob('logs/config/*')) == 0


def test_snapshot_created_on_start_and_stop():
    '''
    Validates that startup.cfg & shutdown.cfg are created upon start & stop.
    '''
    delete_all_snapshots()
    start_torchserve()
    stop_torchserve()
    assert len(glob.glob('logs/config/*startup.cfg')) == 1
    assert len(glob.glob('logs/config/*shutdown.cfg')) == 1


def test_snapshot_created_on_management_api_invoke():
    '''
    Validates that snapshot.cfg is created when management apis are invoked.
    '''
    delete_all_snapshots()
    start_torchserve()
    requests.post('http://127.0.0.1:8081/models?url=https://torchserve.s3.amazonaws.com/mar_files/densenet161.mar')
    time.sleep(10)
    stop_torchserve()
    assert len(glob.glob('logs/config/*snap*.cfg')) == 1

def test_start_from_snapshot():
    '''
    Validates if we can restore state from snapshot.
    '''
    snapshot_cfg = glob.glob('logs/config/*snap*.cfg')[0]
    start_torchserve(snapshot_file=snapshot_cfg)
    response = requests.get('http://127.0.0.1:8081/models/')
    assert json.loads(response.content)['models'][0]['modelName'] == "densenet161"
    stop_torchserve()


def test_start_from_latest():
    '''
    Validates if latest snapshot file is picked if we dont pass snapshot arg explicitly.
    '''
    start_torchserve()
    response = requests.get('http://127.0.0.1:8081/models/')
    assert json.loads(response.content)['models'][0]['modelName'] == "densenet161"
    stop_torchserve()

def test_no_config_snapshots_cli_option():
    '''
    Validates that --no-config-snapshots works as expected.
    '''
    delete_all_snapshots()
    start_torchserve(no_config_snapshots=True)
    stop_torchserve()
    assert len(glob.glob('logs/config/*.cfg')) == 0

def test_start_from_default():
    '''
    Validates that Default config is used if we dont use a config explicitly.
    '''
    delete_all_snapshots()
    start_torchserve()
    response = requests.get('http://127.0.0.1:8081/models/')
    assert len(json.loads(response.content)['models']) == 0
