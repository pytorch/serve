# Regression test cases (test_snapshot.py)
import subprocess
import time
import os
import glob
import requests
import json
from os import path

ROOT_DIR = "/workspace/"
CODEBUILD_WD = path.abspath(path.join(__file__, "../.."))


def start_torchserve(model_store=None, snapshot_file=None, no_config_snapshots=False):
    stop_torchserve()
    cmd = ["torchserve", "--start"]
    model_store = model_store if (model_store != None) else "/workspace/model_store/"
    cmd.extend(["--model-store", "/workspace/model_store/"])
    if (snapshot_file != None):
        cmd.extend(["--ts-config", snapshot_file])
    if (no_config_snapshots):
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


def delete_model_store(model_store=None):
    model_store = model_store if (model_store != None) else "/workspace/model_store/"
    for f in glob.glob(model_store + "/*"):
        os.remove(f)


def test_start_from_default():
    '''
    Validates that Default config is used if we dont use a config explicitly.
    '''
    delete_all_snapshots()
    start_torchserve()
    response = requests.get('http://127.0.0.1:8081/models/')
    assert len(json.loads(response.content)['models']) == 0


def test_logs_created(no_config_snapshots=False):
    stop_torchserve()
    delete_all_snapshots()
    if (no_config_snapshots):
        start_torchserve(no_config_snapshots=True)
    else:
        start_torchserve()
    assert len(glob.glob('logs/access_log.log')) == 1
    assert len(glob.glob('logs/model_log.log')) == 1
    assert len(glob.glob('logs/ts_log.log')) == 1


def test_logs_startup_cfg_created_snapshot_enabled():
    '''
    Validates that access logs are getting created correctly.
    '''
    test_logs_created(no_config_snapshots=False)
    assert len(glob.glob('logs/config/*startup.cfg')) == 1


def test_logs_startup_cfg_created_snapshot_disabled():
    '''
    Validates that access logs are getting created correctly.
    '''
    test_logs_created(no_config_snapshots=True)
    assert len(glob.glob('logs/config/*startup.cfg')) == 0


def test_metrics_created(no_config_snapshots=False):
    stop_torchserve()
    delete_all_snapshots()
    if (no_config_snapshots):
        start_torchserve(no_config_snapshots=True)
    else:
        start_torchserve()
    assert len(glob.glob('logs/model_metrics.log')) == 1
    assert len(glob.glob('logs/ts_metrics.log')) == 1


def test_metrics_startup_cfg_created_snapshot_enabled():
    '''
    Validates that model metrics is getting created.
    '''
    test_metrics_created(no_config_snapshots=False)
    assert len(glob.glob('logs/config/*startup.cfg')) == 1


def test_metrics_startup_cfg_created_snapshot_disabled():
    '''
    Validates that model metrics is getting created.
    '''
    test_metrics_created(no_config_snapshots=True)
    assert len(glob.glob('logs/config/*startup.cfg')) == 0


def validate_config_file(config_file):
    stop_torchserve()
    delete_all_snapshots()
    start_torchserve(snapshot_file=config_file)
    response = requests.get('http://localhost:8080/ping')
    assert json.loads(response.content)['status'] == 'Healthy'
    stop_torchserve()
    delete_all_snapshots()


def validate_ts_config(config_file=None):
    if config_file != None:
        file_contents = []
        try:
            with open(config_file, "r+") as f:
                for line in f:
                    if line.startswith('#') or line in ['\n', '\r\n']:
                        continue  # skip comments
                    line = line.strip()
                    # Check whether it is a key value pair seperated by "=" character
                    assert len(line.split("=")) == 2
        except:
            assert False, "Invalid configuration file"
        else:
            assert True, "Valid config file found"


def test_malformed_ts_config():
    cmd1 = ["cp", "" + CODEBUILD_WD + "/benchmarks/config.properties", ROOT_DIR]
    cmd2 = ["cp", "" + CODEBUILD_WD + "/benchmarks/config.properties",
            ROOT_DIR + "malformed-config.properties"]
    subprocess.run(cmd1)
    subprocess.run(cmd2)
    config_file = "/workspace/config.properties"
    malformed_config_file = "/workspace/malformed-config.properties"
    with open(malformed_config_file, "r+") as f:
        f.writelines(["non-keyvaluepair\n"])
    # First validate well-formed config file
    try:
        conf_file = config_file
        # validate_ts_config(config_file)
        validate_config_file(conf_file)
        # Next validate malformed config file
        conf_file = malformed_config_file
        validate_config_file(conf_file)
#        validate_ts_config(malformed_config_file)
    except:
        validate_ts_config(conf_file)
    finally:
        cmd1 = ["rm", "-rf", config_file]
        cmd2 = ["rm", "-rf", malformed_config_file]
        subprocess.run(cmd1)
        subprocess.run(cmd2)
        stop_torchserve()
        delete_all_snapshots()
        delete_model_store()


def run_LOG_LOCATION_var(custom_path=ROOT_DIR, no_config_snapshots=False):
    delete_all_snapshots()
    if (no_config_snapshots):
        start_torchserve(no_config_snapshots=True)
    else:
        start_torchserve()
    if os.access(custom_path, os.W_OK):
        assert len(glob.glob(custom_path + '/access_log.log')) == 1
        assert len(glob.glob(custom_path + '/model_log.log')) == 1
        assert len(glob.glob(custom_path + '/ts_log.log')) == 1


def test_LOG_LOCATION_var_snapshot_disabled():
    '''
    Validates that access logs are getting created correctly.
    '''
    stop_torchserve()
    os.environ['LOG_LOCATION'] = ROOT_DIR
    run_LOG_LOCATION_var(no_config_snapshots=True)
    stop_torchserve()
    del os.environ['LOG_LOCATION']
    for f in glob.glob(ROOT_DIR + "*.log"):
        os.remove(f)
    delete_model_store()
    # Remove any old snapshots
    delete_all_snapshots()


def test_LOG_LOCATION_var_snapshot_enabled():
    '''
    Validates that access logs are getting created correctly.
    '''
    stop_torchserve()
    os.environ['LOG_LOCATION'] = ROOT_DIR
    run_LOG_LOCATION_var(no_config_snapshots=False)
    requests.post('http://127.0.0.1:8081/models?url=https://torchserve.s3.amazonaws.com/mar_files/'
                  'densenet161.mar')
    time.sleep(10)
    stop_torchserve()
    assert len(glob.glob(ROOT_DIR + 'config/*startup.cfg')) >= 1
    assert len(glob.glob(ROOT_DIR + 'config/*shutdown.cfg')) >= 1
    assert len(glob.glob(ROOT_DIR + 'config/*snap*.cfg')) >= 1
    del os.environ['LOG_LOCATION']
    for f in glob.glob(ROOT_DIR + "*.log"):
        os.remove(f)
    cmd = ["rm", "-rf", ROOT_DIR + 'config']
    subprocess.run(cmd)
    delete_model_store()
    # Remove any old snapshots
    delete_all_snapshots()


def test_async_logging():
    stop_torchserve()
    delete_all_snapshots()
    async_config_file = ROOT_DIR + 'async-log-config.properties'
    with open(async_config_file, "w+") as f:
        f.write("async_logging=true")
    start_torchserve(snapshot_file=async_config_file)
    assert len(glob.glob('logs/access_log.log')) == 1
    assert len(glob.glob('logs/model_log.log')) == 1
    assert len(glob.glob('logs/ts_log.log')) == 1
    stop_torchserve()
    delete_all_snapshots()


def test_async_logging_non_boolean():
    stop_torchserve()
    delete_all_snapshots()
    async_config_file = ROOT_DIR + 'async-log-config.properties'
    with open(async_config_file, "w+") as f:
        f.write("async_logging=2")
    start_torchserve(snapshot_file=async_config_file)
    assert len(glob.glob('logs/access_log.log')) == 1
    assert len(glob.glob('logs/model_log.log')) == 1
    assert len(glob.glob('logs/ts_log.log')) == 1
    stop_torchserve()
    delete_all_snapshots()


def run_METRICS_LOCATION_var(custom_path=ROOT_DIR, no_config_snapshots=False):
    delete_all_snapshots()
    if (no_config_snapshots):
        start_torchserve(no_config_snapshots=True)
    else:
        start_torchserve()
    if os.access(custom_path, os.W_OK):
        assert len(glob.glob(custom_path + '/ts_metrics.log')) == 1
        assert len(glob.glob(custom_path + '/model_metrics.log')) == 1


def test_METRICS_LOCATION_var_snapshot_disabled():
    '''
    Validates that access logs are getting created correctly.
    '''
    stop_torchserve()
    os.environ['METRICS_LOCATION'] = ROOT_DIR
    run_METRICS_LOCATION_var(no_config_snapshots=True)
    stop_torchserve()
    del os.environ['METRICS_LOCATION']
    for f in glob.glob(ROOT_DIR + "*.log"):
        os.remove(f)
    delete_model_store()
    # Remove any old snapshots
    delete_all_snapshots()


def test_METRICS_LOCATION_var_snapshot_enabled():
    '''
    Validates that access logs are getting created correctly.
    '''
    stop_torchserve()
    os.environ['METRICS_LOCATION'] = ROOT_DIR
    run_METRICS_LOCATION_var(no_config_snapshots=False)
    requests.post('http://127.0.0.1:8081/models?url=https://torchserve.s3.amazonaws.com/mar_files/'
                  'densenet161.mar')
    time.sleep(10)
    stop_torchserve()
    assert len(glob.glob('logs/config/*startup.cfg')) >= 1
    assert len(glob.glob('logs/config/*shutdown.cfg')) >= 1
    assert len(glob.glob('logs/config/*snap*.cfg')) >= 1
    del os.environ['METRICS_LOCATION']
    for f in glob.glob(ROOT_DIR + "*.log"):
        os.remove(f)
    delete_all_snapshots()
    delete_model_store()


def test_LOG_LOCATION_METRICS_LOCATION_vars_snapshot_enabled():
    delete_all_snapshots()
    delete_model_store()
    stop_torchserve()
    os.environ['LOG_LOCATION'] = ROOT_DIR
    os.environ['METRICS_LOCATION'] = ROOT_DIR
    run_LOG_LOCATION_var(no_config_snapshots=False)
    run_METRICS_LOCATION_var(no_config_snapshots=False)
    requests.post('http://127.0.0.1:8081/models?url=https://torchserve.s3.amazonaws.com/mar_files/'
                  'densenet161.mar')
    time.sleep(10)
    stop_torchserve()
    assert len(glob.glob(ROOT_DIR + 'config/*startup.cfg')) >= 1
    assert len(glob.glob(ROOT_DIR + 'config/*shutdown.cfg')) >= 1
    assert len(glob.glob(ROOT_DIR + 'config/*snap*.cfg')) >= 1
    del os.environ['LOG_LOCATION']
    del os.environ['METRICS_LOCATION']
    for f in glob.glob(ROOT_DIR + "*.log"):
        os.remove(f)
    cmd = ["rm", "-rf", ROOT_DIR + 'config']
    subprocess.run(cmd)
    delete_model_store()
    # Remove any old snapshots
    delete_all_snapshots()


def test_LOG_LOCATION_var_snapshot_disabled_custom_path_read_only():
    '''
    Validates that access logs are getting created correctly.
    '''
    stop_torchserve()
    # First remove existing logs otherwise it may be a false positive case
    for f in glob.glob('logs/*.log'):
        os.remove(f)
    RDONLY_DIR = '/workspace/rdonly_dir'
    os.environ['LOG_LOCATION'] = RDONLY_DIR
    try:
        run_LOG_LOCATION_var(custom_path=RDONLY_DIR, no_config_snapshots=True)
        assert len(glob.glob('logs/access_log.log')) == 1
        assert len(glob.glob('logs/model_log.log')) == 1
        assert len(glob.glob('logs/ts_log.log')) == 1
        assert len(glob.glob('logs/model_metrics.log')) == 1
        assert len(glob.glob('logs/ts_metrics.log')) == 1
    finally:
        stop_torchserve()
        del os.environ['LOG_LOCATION']
        delete_model_store()
        # Remove any old snapshots
        delete_all_snapshots()


def test_METRICS_LOCATION_var_snapshot_enabled_rdonly_dir():
    '''
    Validates that access logs are getting created correctly.
    '''
    stop_torchserve()
    # First remove existing logs otherwise it may be a false positive case
    for f in glob.glob('logs/*.log'):
        os.remove(f)
    RDONLY_DIR = '/workspace/rdonly_dir'
    os.environ['METRICS_LOCATION'] = RDONLY_DIR
    try:
        run_METRICS_LOCATION_var(custom_path=RDONLY_DIR, no_config_snapshots=False)
        requests.post('http://127.0.0.1:8081/models?url=https://torchserve.s3.amazonaws.com/'
                      'mar_files/densenet161.mar')
        time.sleep(10)
        assert len(glob.glob('logs/access_log.log')) == 1
        assert len(glob.glob('logs/model_log.log')) == 1
        assert len(glob.glob('logs/ts_log.log')) == 1
        assert len(glob.glob('logs/model_metrics.log')) == 1
        assert len(glob.glob('logs/ts_metrics.log')) == 1
        assert len(glob.glob('logs/config/*snap*.cfg')) >= 1
    finally:
        stop_torchserve()
        del os.environ['METRICS_LOCATION']
        delete_all_snapshots()
        delete_model_store()
