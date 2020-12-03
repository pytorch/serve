import platform
import shutil
import subprocess
import os
import time
from concurrent.futures import thread
from os import path
import glob
import requests
import json
import test_utils

NUM_STARTUP_CFG = 0


def setup_module(module):
    test_utils.torchserve_cleanup()
    response = requests.get("https://torchserve.pytorch.org/mar_files/densenet161.mar", allow_redirects=True)
    open(test_utils.MODEL_STORE + "/densenet161.mar", 'wb').write(response.content)


def teardown_module(module):
    test_utils.torchserve_cleanup()


def logs_created(no_config_snapshots=False):
    test_utils.start_torchserve(no_config_snapshots=no_config_snapshots)
    assert len(glob.glob('logs/access_log.log')) == 1
    assert len(glob.glob('logs/model_log.log')) == 1
    assert len(glob.glob('logs/ts_log.log')) == 1

def validate_metrics_created(no_config_snapshots=False):
    test_utils.delete_all_snapshots()
    global NUM_STARTUP_CFG
    # Reset NUM_STARTUP_CFG as we are deleting snapshots in the previous step
    NUM_STARTUP_CFG = 0
    test_utils.start_torchserve(no_config_snapshots=no_config_snapshots)
    if not no_config_snapshots:
        NUM_STARTUP_CFG += 1

    assert len(glob.glob('logs/model_metrics.log')) == 1
    assert len(glob.glob('logs/ts_metrics.log')) == 1


def run_log_location_var(custom_path=test_utils.ROOT_DIR, no_config_snapshots=False):
    test_utils.delete_all_snapshots()
    test_utils.start_torchserve(no_config_snapshots=no_config_snapshots)

    # This check warrants that we are not accidentally monitoring a readonly logs/snapshot directory
    if os.access(custom_path, os.W_OK):
        assert len(glob.glob(custom_path + '/access_log.log')) == 1
        assert len(glob.glob(custom_path + '/model_log.log')) == 1
        assert len(glob.glob(custom_path + '/ts_log.log')) == 1


def test_logs_created():
    logs_created()
    global NUM_STARTUP_CFG
    NUM_STARTUP_CFG = len(glob.glob('logs/config/*startup.cfg'))


def test_logs_startup_cfg_created_snapshot_enabled():
    """
    Validates that access logs are getting created correctly.
    """
    logs_created(no_config_snapshots=False)
    global NUM_STARTUP_CFG
    assert len(glob.glob('logs/config/*startup.cfg')) == NUM_STARTUP_CFG + 1
    NUM_STARTUP_CFG += 1


def test_logs_startup_cfg_created_snapshot_disabled():
    """
    Validates that access logs are getting created correctly.
    """
    logs_created(no_config_snapshots=True)
    global NUM_STARTUP_CFG
    assert len(glob.glob('logs/config/*startup.cfg')) == NUM_STARTUP_CFG


def test_metrics_startup_cfg_created_snapshot_enabled():
    """
    Validates that model metrics is getting created with snapshot enabled.
    """
    validate_metrics_created(no_config_snapshots=False)
    assert len(glob.glob('logs/config/*startup.cfg')) == NUM_STARTUP_CFG


def test_metrics_startup_cfg_created_snapshot_disabled():
    """
    Validates that model metrics is getting created with snapshot disabled.
    """
    validate_metrics_created(no_config_snapshots=True)
    assert len(glob.glob('logs/config/*startup.cfg')) == 0


def test_log_location_var_snapshot_disabled():
    """
    Validates that non metrics logs get saved in directory configured via LOG_LOCATION
    environment variable.
    """
    # We stop torchserve here so that we can set LOG_LOCATION in environment variable and rerun torchserve
    test_utils.stop_torchserve()
    os.environ['LOG_LOCATION'] = test_utils.ROOT_DIR
    run_log_location_var(no_config_snapshots=True)
    # We stop torchserve again here so that we can remove the LOG_LOCATION setting from environment variable
    test_utils.stop_torchserve()
    del os.environ['LOG_LOCATION']
    for f in glob.glob(path.join(test_utils.ROOT_DIR, "*.log")):
        print("-------------Deleting " + f)
        os.remove(f)
    # Remove any old snapshots
    test_utils.delete_all_snapshots()


def test_log_location_var_snapshot_enabled():
    """
    Validates that non metrics logs get saved in directory configured via LOG_LOCATION
    environment variable.
    """
    # We stop torchserve here so that we can set LOG_LOCATION in environment variable and rerun torchserve
    test_utils.stop_torchserve()
    os.environ['LOG_LOCATION'] = test_utils.ROOT_DIR
    run_log_location_var(no_config_snapshots=False)
    requests.post('http://127.0.0.1:8081/models?url=densenet161.mar')
    # We stop torchserve again here so that we can remove the LOG_LOCATION setting from environment variable
    test_utils.stop_torchserve()
    print('Waiting to stop')
    time.sleep(15)
    del os.environ['LOG_LOCATION']

    # In case of snapshot enabled, we get these three config files additionally in the custom directory
    assert len(glob.glob(path.join(test_utils.ROOT_DIR,'config/*startup.cfg'))) >= 1
    if platform.system() != "Windows":
        assert len(glob.glob(path.join(test_utils.ROOT_DIR, 'config/*shutdown.cfg'))) >= 1
    assert len(glob.glob(path.join(test_utils.ROOT_DIR, 'config/*snap*.cfg'))) >= 1
    for f in glob.glob(path.join(test_utils.ROOT_DIR, "*.log")):
        print("-------------Deleting "+f)
        os.remove(f)

    shutil.rmtree(path.join(test_utils.ROOT_DIR, 'config'))

    # Remove any old snapshots
    test_utils.delete_all_snapshots()


def test_async_logging():
    """Validates that we can use async_logging flag while starting Torchserve"""
    # Need to stop torchserve as we need to check if log files get generated with 'aysnc_logging' flag
    test_utils.stop_torchserve()
    for f in glob.glob("logs/*.log"):
        os.remove(f)
    # delete_all_snapshots()
    async_config_file = test_utils.ROOT_DIR + 'async-log-config.properties'
    with open(async_config_file, "w+") as f:
        f.write("async_logging=true")
    test_utils.start_torchserve(snapshot_file=async_config_file)
    assert len(glob.glob('logs/access_log.log')) == 1
    assert len(glob.glob('logs/model_log.log')) == 1
    assert len(glob.glob('logs/ts_log.log')) == 1


def test_async_logging_non_boolean():
    '''Validates that Torchserve uses default value for async_logging flag
    if we assign a non boolean value to this flag'''
    test_utils.stop_torchserve()
    for f in glob.glob("logs/*.log"):
        os.remove(f)
    # delete_all_snapshots()
    async_config_file = test_utils.ROOT_DIR + 'async-log-config.properties'
    with open(async_config_file, "w+") as f:
        f.write("async_logging=2")
    test_utils.start_torchserve(snapshot_file=async_config_file)
    assert len(glob.glob('logs/access_log.log')) == 1
    assert len(glob.glob('logs/model_log.log')) == 1
    assert len(glob.glob('logs/ts_log.log')) == 1
    test_utils.stop_torchserve()


def run_metrics_location_var(custom_path=test_utils.ROOT_DIR, no_config_snapshots=False):
    test_utils.delete_all_snapshots()
    test_utils.start_torchserve(no_config_snapshots=no_config_snapshots)

    if os.access(custom_path, os.W_OK):
        assert len(glob.glob(custom_path + '/ts_metrics.log')) == 1
        assert len(glob.glob(custom_path + '/model_metrics.log')) == 1


def test_metrics_location_var_snapshot_disabled():
    """
    Validates that metrics related logs get saved in directory configured via METRICS_LOCATION
    environment variable.
    """
    # We stop torchserve here so that we can set METRICS_LOCATION in environment variable and
    # restart torchserve
    test_utils.stop_torchserve()
    os.environ['METRICS_LOCATION'] = test_utils.ROOT_DIR
    run_metrics_location_var(no_config_snapshots=True)
    # We stop torchserve again here so that we can remove the METRICS_LOCATION setting
    # from environment variable
    test_utils.stop_torchserve()
    del os.environ['METRICS_LOCATION']
    for f in glob.glob(test_utils.ROOT_DIR + "*.log"):
        os.remove(f)
    # Remove any old snapshots
    test_utils.delete_all_snapshots()


def test_metrics_location_var_snapshot_enabled():
    """
    Validates that metrics related logs get saved in directory configured via METRICS_LOCATION
    environment variable.
    """
    # We stop torchserve here so that we can set METRICS_LOCATION in environment variable and
    # restart torchserve
    test_utils.stop_torchserve()
    os.environ['METRICS_LOCATION'] = test_utils.ROOT_DIR
    run_metrics_location_var(no_config_snapshots=False)
    requests.post('http://127.0.0.1:8081/models?url=densenet161.mar')
    # We stop torchserve again here so that we can remove the METRICS_LOCATION setting
    # from environment variable
    test_utils.stop_torchserve()
    del os.environ['METRICS_LOCATION']
    # In case of snapshot enabled, we get these three config files additionally in the custom directory
    assert len(glob.glob('logs/config/*startup.cfg')) >= 1
    if platform.system() != "Windows":
        assert len(glob.glob('logs/config/*shutdown.cfg')) >= 1
    assert len(glob.glob('logs/config/*snap*.cfg')) >= 1
    for f in glob.glob(test_utils.ROOT_DIR + "*.log"):
        os.remove(f)


def test_log_location_and_metric_location_vars_snapshot_enabled():
    """
    Validates that metrics & non metrics related logs get saved in directory configured as per
     METRICS_LOCATION & LOG_LOCATION environment variables with snaphsot enabled.
    """
    test_utils.stop_torchserve()
    test_utils.delete_all_snapshots()
    os.environ['LOG_LOCATION'] = test_utils.ROOT_DIR
    os.environ['METRICS_LOCATION'] = test_utils.ROOT_DIR
    run_log_location_var(no_config_snapshots=False)
    run_metrics_location_var(no_config_snapshots=False)
    requests.post('http://127.0.0.1:8081/models?url=densenet161.mar')
    # We stop torchserve again here so that we can remove the LOG_LOCATION & METRICS_LOCATION
    # setting from environment variable
    test_utils.stop_torchserve()
    del os.environ['LOG_LOCATION']
    del os.environ['METRICS_LOCATION']
    assert len(glob.glob(test_utils.ROOT_DIR + 'config/*startup.cfg')) >= 1
    if platform.system() != "Windows":
        assert len(glob.glob(test_utils.ROOT_DIR + 'config/*shutdown.cfg')) >= 1
    assert len(glob.glob(test_utils.ROOT_DIR + 'config/*snap*.cfg')) >= 1
    for f in glob.glob(test_utils.ROOT_DIR + "*.log"):
        os.remove(f)

    shutil.rmtree(path.join(test_utils.ROOT_DIR, 'config'))


def test_log_location_var_snapshot_disabled_custom_path_read_only():
    """
    Validates that we should not be able to create non metrics related logs if the directory configured
    via 'LOG_LOCATION' is a read only directory.
    """
    # Torchserve cleanup is required here as we are going to set 'LOG_LOCATION' environment variable
    test_utils.stop_torchserve()
    test_utils.delete_all_snapshots()
    # First remove existing logs otherwise it may be a false positive case
    for f in glob.glob('logs/*.log'):
        os.remove(f)
    RDONLY_DIR = path.join(test_utils.ROOT_DIR, 'rdonly_dir')
    os.environ['LOG_LOCATION'] = RDONLY_DIR
    try:
        run_log_location_var(custom_path=RDONLY_DIR, no_config_snapshots=True)
        assert len(glob.glob(RDONLY_DIR + '/logs/access_log.log')) == 0
        assert len(glob.glob(RDONLY_DIR + '/logs/model_log.log')) == 0
        assert len(glob.glob(RDONLY_DIR + '/logs/ts_log.log')) == 0
        assert len(glob.glob('logs/model_metrics.log')) == 1
        assert len(glob.glob('logs/ts_metrics.log')) == 1
    finally:
        del os.environ['LOG_LOCATION']


def test_metrics_location_var_snapshot_enabled_rdonly_dir():
    """
    Validates that we should not be able to create metrics related logs if the directory configured
    via 'METRICS_LOCATION' is a read only directory.
    """
    # Torchserve cleanup is required here as we are going to set 'LOG_LOCATION' environment variable
    test_utils.stop_torchserve()
    test_utils.delete_all_snapshots()
    # First remove existing logs otherwise it may be a false positive case
    for f in glob.glob('logs/*.log'):
        os.remove(f)
    RDONLY_DIR = path.join(test_utils.ROOT_DIR,'rdonly_dir')
    os.environ['METRICS_LOCATION'] = RDONLY_DIR
    try:
        run_metrics_location_var(custom_path=RDONLY_DIR, no_config_snapshots=False)
        requests.post('http://127.0.0.1:8081/models?url=densenet161.mar')
        assert len(glob.glob('logs/access_log.log')) == 1
        assert len(glob.glob('logs/model_log.log')) == 1
        assert len(glob.glob('logs/ts_log.log')) == 1
        assert len(glob.glob('logs/config/*snap*.cfg')) == 1
        assert len(glob.glob(RDONLY_DIR + '/logs/model_metrics.log')) == 0
        assert len(glob.glob(RDONLY_DIR + '/logs/ts_metrics.log')) == 0
    finally:
        del os.environ['METRICS_LOCATION']
