#Regression test cases (test_snapshot.py)
import subprocess
import time
import os
import glob
import requests
import json
import pandas as pd

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


def test_start_from_default():
    '''
    Validates that Default config is used if we dont use a config explicitly.
    '''
    delete_all_snapshots()
    start_torchserve()
    response = requests.get('http://127.0.0.1:8081/models/')
    assert len(json.loads(response.content)['models']) == 0


def test_access_log_created():
    '''
    Validates that access logs are getting created correctly.
    '''
    stop_torchserve()
    test_start_from_default()
    assert len(glob.glob('logs/access_log.log')) == 1

def test_model_log_created():
    '''
    Validates that model logs are getting created correctly.
    '''
    stop_torchserve()
    test_start_from_default()
    assert len(glob.glob('logs/model_log.log')) == 1

def test_ts_log_created():
    '''
    Validates that ts logs are getting created correctly.
    '''
    stop_torchserve()
    test_start_from_default()
    assert len(glob.glob('logs/ts_log.log')) == 1

def test_model_metrics_created():
    '''
    Validates that model metrics is getting created.
    '''
    stop_torchserve()
    test_start_from_default()
    assert len(glob.glob('logs/model_metrics.log')) == 1

def test_ts_metrics_created():
    '''
    Validates that ts metrics is getting created correctly.
    '''
    stop_torchserve()
    test_start_from_default()
    assert len(glob.glob('logs/ts_metrics.log')) == 1


def validate_ts_config(config_file=None):
    if config_file !=None:
        file_contents = []
        try:
            with open(config_file, "r+") as f:
                for line in f:
                    if line.startswith('#'):
                        continue  # skip comments
                    line = line.strip()
                    #Check whether it is a key value pair seperated by "=" character
                    assert len(line.split("=")) == 2
        except:
            assert False, "Invalid configuration file"
        else:
            assert True, "Valid config file found"


def test_malformed_ts_config():
    config_file = "/workspace/config.properties"
    malformed_config_file = "/workspace/malformed_config.properties"
    # First validate well-formed config file
    validate_ts_config(config_file)
    # Next validate malformed config file
    validate_ts_config(malformed_config_file)


def test_test1():
    pass
