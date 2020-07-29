import subprocess
import time
import os
import glob
import requests
import json
from os import path

ROOT_DIR = "/workspace/"
MODEL_STORE = ROOT_DIR + "model_store/"
CODEBUILD_WD = path.abspath(path.join(__file__, "../../.."))
NUM_STARTUP_CFG = 0


def start_torchserve(model_store=None, snapshot_file=None, no_config_snapshots=False):
    stop_torchserve()
    cmd = ["torchserve", "--start"]
    model_store = model_store if (model_store != None) else MODEL_STORE
    cmd.extend(["--model-store", model_store])
    if (snapshot_file != None):
        cmd.extend(["--ts-config", snapshot_file])
    if (no_config_snapshots):
        cmd.extend(["--no-config-snapshots"])
    subprocess.run(cmd)
    time.sleep(10)


def stop_torchserve():
    subprocess.run(["torchserve", "--stop"])
    time.sleep(5)


def delete_all_snapshots(custom_dir=''):
    for f in glob.glob(custom_dir + 'logs/config/*'):
        os.remove(f)
    assert len(glob.glob(custom_dir + 'logs/config/*')) == 0


def delete_model_store(model_store=None):
    '''Removes all model mar files from model store'''
    model_store = model_store if (model_store != None) else MODEL_STORE
    for f in glob.glob(model_store + "/*.mar"):
        os.remove(f)


def torchserve_cleanup():
    stop_torchserve()
    delete_model_store()
    delete_all_snapshots()


def test_cleanup():
    torchserve_cleanup()


def logs_created(no_config_snapshots=False):
    if (no_config_snapshots):
        start_torchserve(no_config_snapshots=True)
    else:
        start_torchserve()
    assert len(glob.glob('logs/access_log.log')) == 1
    assert len(glob.glob('logs/model_log.log')) == 1
    assert len(glob.glob('logs/ts_log.log')) == 1


def test_logs_created(no_config_snapshots=False):
    logs_created()
    global NUM_STARTUP_CFG
    NUM_STARTUP_CFG = len(glob.glob('logs/config/*startup.cfg'))


def test_logs_startup_cfg_created_snapshot_enabled():
    '''
    Validates that access logs are getting created correctly.
    '''
    logs_created(no_config_snapshots=False)
    global NUM_STARTUP_CFG
    assert len(glob.glob('logs/config/*startup.cfg')) == NUM_STARTUP_CFG + 1
    NUM_STARTUP_CFG += 1


def test_logs_startup_cfg_created_snapshot_disabled():
    '''
    Validates that access logs are getting created correctly.
    '''
    logs_created(no_config_snapshots=True)
    global NUM_STARTUP_CFG
    assert len(glob.glob('logs/config/*startup.cfg')) == NUM_STARTUP_CFG


def test_metrics_created(no_config_snapshots=False):
    delete_all_snapshots()
    global NUM_STARTUP_CFG
    #Reset NUM_STARTUP_CFG as we are deleting snapshots in the previous step
    NUM_STARTUP_CFG = 0
    if (no_config_snapshots):
        start_torchserve(no_config_snapshots=True)
    else:
        start_torchserve()
        NUM_STARTUP_CFG += 1
    assert len(glob.glob('logs/model_metrics.log')) == 1
    assert len(glob.glob('logs/ts_metrics.log')) == 1


def test_metrics_startup_cfg_created_snapshot_enabled():
    '''
    Validates that model metrics is getting created with snapshot enabled.
    '''
    test_metrics_created(no_config_snapshots=False)
    assert len(glob.glob('logs/config/*startup.cfg')) == NUM_STARTUP_CFG


def test_metrics_startup_cfg_created_snapshot_disabled():
    '''
    Validates that model metrics is getting created with snapshot disabled.
    '''
    test_metrics_created(no_config_snapshots=True)
    assert len(glob.glob('logs/config/*startup.cfg')) == 0


def validate_config_file(config_file):
    start_torchserve(snapshot_file=config_file)
    response = requests.get('http://localhost:8080/ping')
    assert json.loads(response.content)['status'] == 'Healthy'


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
    '''Validates that Torchserve validates the config file parameters correctly and
    ignores any unknown key-value parameters and starts successfully!!'''
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
        validate_config_file(conf_file)
        # Next validate malformed config file
        conf_file = malformed_config_file
        validate_config_file(conf_file)
    except:
        validate_ts_config(conf_file)
    finally:
        cmd1 = ["rm", "-rf", config_file]
        cmd2 = ["rm", "-rf", malformed_config_file]
        subprocess.run(cmd1)
        subprocess.run(cmd2)
        delete_all_snapshots()
        delete_model_store()


def run_LOG_LOCATION_var(custom_path=ROOT_DIR, no_config_snapshots=False):
    delete_all_snapshots()
    if (no_config_snapshots):
        start_torchserve(no_config_snapshots=True)
    else:
        start_torchserve()
    #This check warrants that we are not accidentally monitoring a readonly logs/snapshot directory
    if os.access(custom_path, os.W_OK):
        assert len(glob.glob(custom_path + '/access_log.log')) == 1
        assert len(glob.glob(custom_path + '/model_log.log')) == 1
        assert len(glob.glob(custom_path + '/ts_log.log')) == 1


def test_LOG_LOCATION_var_snapshot_disabled():
    '''
    Validates that non metrics logs get saved in directory configured via LOG_LOCATION
    environment variable.
    '''
    # We stop torchserve here so that we can set LOG_LOCATION in environment variable and rerun torchserve
    stop_torchserve()
    os.environ['LOG_LOCATION'] = ROOT_DIR
    run_LOG_LOCATION_var(no_config_snapshots=True)
    # We stop torchserve again here so that we can remove the LOG_LOCATION setting from environment variable
    stop_torchserve()
    del os.environ['LOG_LOCATION']
    for f in glob.glob(ROOT_DIR + "*.log"):
        os.remove(f)
    delete_model_store()
    # Remove any old snapshots
    delete_all_snapshots()


def test_LOG_LOCATION_var_snapshot_enabled():
    '''
    Validates that non metrics logs get saved in directory configured via LOG_LOCATION
    environment variable.
    '''
    # We stop torchserve here so that we can set LOG_LOCATION in environment variable and rerun torchserve
    stop_torchserve()
    os.environ['LOG_LOCATION'] = ROOT_DIR
    run_LOG_LOCATION_var(no_config_snapshots=False)
    requests.post('http://127.0.0.1:8081/models?url=https://torchserve.s3.amazonaws.com/mar_files/'
                  'densenet161.mar')
    time.sleep(10)
    # We stop torchserve again here so that we can remove the LOG_LOCATION setting from environment variable
    stop_torchserve()
    del os.environ['LOG_LOCATION']
    #In case of snapshot enabled, we get these three config files additionally in the custom directory
    assert len(glob.glob(ROOT_DIR + 'config/*startup.cfg')) >= 1
    assert len(glob.glob(ROOT_DIR + 'config/*shutdown.cfg')) >= 1
    assert len(glob.glob(ROOT_DIR + 'config/*snap*.cfg')) >= 1
    for f in glob.glob(ROOT_DIR + "*.log"):
        os.remove(f)
    cmd = ["rm", "-rf", ROOT_DIR + 'config']
    subprocess.run(cmd)
    delete_model_store()
    # Remove any old snapshots
    delete_all_snapshots()


def test_async_logging():
    '''Validates that we can use async_logging flag while starting Torchserve'''
    # Need to stop torchserve as we need to check if log files get generated with 'aysnc_logging' flag
    stop_torchserve()
    for f in glob.glob("logs/*.log"):
        os.remove(f)
    # delete_all_snapshots()
    async_config_file = ROOT_DIR + 'async-log-config.properties'
    with open(async_config_file, "w+") as f:
        f.write("async_logging=true")
    start_torchserve(snapshot_file=async_config_file)
    assert len(glob.glob('logs/access_log.log')) == 1
    assert len(glob.glob('logs/model_log.log')) == 1
    assert len(glob.glob('logs/ts_log.log')) == 1


def test_async_logging_non_boolean():
    '''Validates that Torchserve uses default value for async_logging flag
    if we assign a non boolean value to this flag'''
    stop_torchserve()
    for f in glob.glob("logs/*.log"):
        os.remove(f)
    # delete_all_snapshots()
    async_config_file = ROOT_DIR + 'async-log-config.properties'
    with open(async_config_file, "w+") as f:
        f.write("async_logging=2")
    start_torchserve(snapshot_file=async_config_file)
    assert len(glob.glob('logs/access_log.log')) == 1
    assert len(glob.glob('logs/model_log.log')) == 1
    assert len(glob.glob('logs/ts_log.log')) == 1
    stop_torchserve()
    # delete_all_snapshots()


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
    Validates that metrics related logs get saved in directory configured via METRICS_LOCATION
    environment variable.
    '''
    # We stop torchserve here so that we can set METRICS_LOCATION in environment variable and
    # restart torchserve
    stop_torchserve()
    os.environ['METRICS_LOCATION'] = ROOT_DIR
    run_METRICS_LOCATION_var(no_config_snapshots=True)
    # We stop torchserve again here so that we can remove the METRICS_LOCATION setting
    # from environment variable
    stop_torchserve()
    del os.environ['METRICS_LOCATION']
    for f in glob.glob(ROOT_DIR + "*.log"):
        os.remove(f)
    delete_model_store()
    # Remove any old snapshots
    delete_all_snapshots()


def test_METRICS_LOCATION_var_snapshot_enabled():
    '''
    Validates that metrics related logs get saved in directory configured via METRICS_LOCATION
    environment variable.
    '''
    # We stop torchserve here so that we can set METRICS_LOCATION in environment variable and
    # restart torchserve
    stop_torchserve()
    os.environ['METRICS_LOCATION'] = ROOT_DIR
    run_METRICS_LOCATION_var(no_config_snapshots=False)
    requests.post('http://127.0.0.1:8081/models?url=https://torchserve.s3.amazonaws.com/mar_files/'
                  'densenet161.mar')
    time.sleep(10)
    # We stop torchserve again here so that we can remove the METRICS_LOCATION setting
    # from environment variable
    stop_torchserve()
    del os.environ['METRICS_LOCATION']
    # In case of snapshot enabled, we get these three config files additionally in the custom directory
    assert len(glob.glob('logs/config/*startup.cfg')) >= 1
    assert len(glob.glob('logs/config/*shutdown.cfg')) >= 1
    assert len(glob.glob('logs/config/*snap*.cfg')) >= 1
    for f in glob.glob(ROOT_DIR + "*.log"):
        os.remove(f)


def test_LOG_LOCATION_and_METRICS_LOCATION_vars_snapshot_enabled():
    '''
    Validates that metrics & non metrics related logs get saved in directory configured as per
     METRICS_LOCATION & LOG_LOCATION environment variables with snaphsot enabled.
    '''
    torchserve_cleanup()
    # delete_all_snapshots()
    # delete_model_store()
    # stop_torchserve()
    os.environ['LOG_LOCATION'] = ROOT_DIR
    os.environ['METRICS_LOCATION'] = ROOT_DIR
    run_LOG_LOCATION_var(no_config_snapshots=False)
    run_METRICS_LOCATION_var(no_config_snapshots=False)
    requests.post('http://127.0.0.1:8081/models?url=https://torchserve.s3.amazonaws.com/mar_files/'
                  'densenet161.mar')
    time.sleep(10)
    # We stop torchserve again here so that we can remove the LOG_LOCATION & METRICS_LOCATION
    # setting from environment variable
    stop_torchserve()
    del os.environ['LOG_LOCATION']
    del os.environ['METRICS_LOCATION']
    assert len(glob.glob(ROOT_DIR + 'config/*startup.cfg')) >= 1
    assert len(glob.glob(ROOT_DIR + 'config/*shutdown.cfg')) >= 1
    assert len(glob.glob(ROOT_DIR + 'config/*snap*.cfg')) >= 1
    for f in glob.glob(ROOT_DIR + "*.log"):
        os.remove(f)
    cmd = ["rm", "-rf", ROOT_DIR + 'config']
    subprocess.run(cmd)
    # delete_model_store()
    # # Remove any old snapshots
    # delete_all_snapshots()


def test_LOG_LOCATION_var_snapshot_disabled_custom_path_read_only():
    '''
    Validates that we should not be able to create non metrics related logs if the directory configured
    via 'LOG_LOCATION' is a read only directory.
    '''
    # Torchserve cleanup is required here as we are going to set 'LOG_LOCATION' environment variable
    torchserve_cleanup()
    # stop_torchserve()
    # First remove existing logs otherwise it may be a false positive case
    for f in glob.glob('logs/*.log'):
        os.remove(f)
    RDONLY_DIR = '/workspace/rdonly_dir'
    os.environ['LOG_LOCATION'] = RDONLY_DIR
    try:
        run_LOG_LOCATION_var(custom_path=RDONLY_DIR, no_config_snapshots=True)
        assert len(glob.glob(RDONLY_DIR+'/logs/access_log.log')) == 0
        assert len(glob.glob(RDONLY_DIR+'/logs/model_log.log')) == 0
        assert len(glob.glob(RDONLY_DIR+'/logs/ts_log.log')) == 0
        assert len(glob.glob('logs/model_metrics.log')) == 1
        assert len(glob.glob('logs/ts_metrics.log')) == 1
    finally:
        del os.environ['LOG_LOCATION']
        # torchserve_cleanup()


def test_METRICS_LOCATION_var_snapshot_enabled_rdonly_dir():
    '''
    Validates that we should not be able to create metrics related logs if the directory configured
    via 'METRICS_LOCATION' is a read only directory.
    '''
    # Torchserve cleanup is required here as we are going to set 'LOG_LOCATION' environment variable
    torchserve_cleanup()
    # stop_torchserve()
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
        assert len(glob.glob('logs/config/*snap*.cfg')) == 1
        assert len(glob.glob(RDONLY_DIR+'/logs/model_metrics.log')) == 0
        assert len(glob.glob(RDONLY_DIR+'/logs/ts_metrics.log')) == 0
    finally:
        del os.environ['METRICS_LOCATION']
        #Final cleanup
        torchserve_cleanup()