import subprocess
import time
import os
import glob
import requests
import json


def start_torchserve(model_store=None, snapshot_file=None, no_config_snapshots=False):
    stop_torchserve()
    cmd = ["torchserve", "--start"]
    model_store = model_store if (model_store != None) else "/workspace/model_store/"
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


def delete_all_snapshots():
    for f in glob.glob('logs/config/*'):
        os.remove(f)
    assert len(glob.glob('logs/config/*')) == 0


def delete_mar_file_from_model_store(model_store=None, model_mar=None):
    model_store = model_store if (model_store != None) else "/workspace/model_store/"
    if model_mar != None:
        for f in glob.glob(model_store + "/" + model_mar + "*"):
            os.remove(f)


def replace_mar_file_with_dummy_mar_in_model_store(model_store=None, model_mar=None):
    model_store = model_store if (model_store != None) else "/workspace/model_store/"
    if model_mar != None:
        myfilepath = model_store + "/" + model_mar
        if os.path.exists(model_store + "/" + model_mar):
            os.remove(myfilepath)
            with open(myfilepath, "w+") as f:
                f.write("junk data")


def delete_model_store(model_store=None):
    '''Removes all model mar files from model store'''
    model_store = model_store if (model_store != None) else "/workspace/model_store/"
    for f in glob.glob(model_store + "/*.mar"):
        os.remove(f)


def torchserve_cleanup():
    stop_torchserve()
    delete_model_store()
    delete_all_snapshots()


def test_cleanup():
    torchserve_cleanup()

def test_snapshot_created_on_start_and_stop():
    '''
    Validates that startup .cfg & shutdown.cfg are created upon start & stop.
    '''
    delete_all_snapshots()
    start_torchserve()
    stop_torchserve()
    assert len(glob.glob('logs/config/*startup.cfg')) == 1
    assert len(glob.glob('logs/config/*shutdown.cfg')) == 1


def snapshot_created_on_management_api_invoke(model_mar="densenet161.mar"):
    delete_all_snapshots()
    start_torchserve()
    requests.post('http://127.0.0.1:8081/models?url=https://torchserve.s3.amazonaws.com/mar_files/'
                  + model_mar)
    time.sleep(10)
    stop_torchserve()


def test_snapshot_created_on_management_api_invoke():
    '''
    Validates that snapshot.cfg is created when management apis are invoked.
    '''
    snapshot_created_on_management_api_invoke()
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


def test_start_from_read_only_snapshot():
    '''
    Validates if we can start and restore Torchserve state using a read-only snapshot.
    '''
    snapshot_cfg = glob.glob('logs/config/*snap*.cfg')[0]
    file_status = os.stat(snapshot_cfg)
    os.chmod(snapshot_cfg, 0o444)
    start_torchserve(snapshot_file=snapshot_cfg)
    os.chmod(snapshot_cfg, (file_status.st_mode & 0o777))
    try:
        response = requests.get('http://127.0.0.1:8081/models/')
    except:
        assert False, "Something is not right!! Failed to start Torchserve using Read Only Snapshot!!"
    else:
        assert True, "Successfully started and restored Torchserve state using a Read Only Snapshot"


def test_no_config_snapshots_cli_option():
    '''
    Validates that --no-config-snapshots works as expected.
    '''
    # Required to stop torchserve here so that all config files gets deleted
    stop_torchserve()
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


def test_start_from_non_existing_snapshot():
    '''
    Validates that Torchserve should fail to start when we pass a non-existent snapshot
     as an input snapshot while starting Torchserve.
    '''
    stop_torchserve()
    start_torchserve(snapshot_file="logs/config/junk-snapshot.cfg")
    try:
        response = requests.get('http://127.0.0.1:8081/models/')
    except:
        assert True, "Failed to start Torchserve using a Non Existing Snapshot"
    else:
        assert False, "Something is not right!! Successfully started Torchserve " \
                      "using Non Existing Snapshot File!!"


def test_torchserve_init_with_non_existent_model_store():
    '''Validates that Torchserve fails to start if the model store directory is non existent '''

    start_torchserve(model_store="/invalid_model_store", snapshot_file=None, no_config_snapshots=True)
    try:
        response = requests.get('http://127.0.0.1:8081/models/')
    except:
        assert True, "Failed to start Torchserve using non existent model-store directory"
    else:
        assert False, "Something is not right!! Successfully started Torchserve " \
                      "using non existent directory!!"
    finally:
        delete_model_store()
        delete_all_snapshots()


def test_restart_torchserve_with_last_snapshot_with_model_mar_removed():
    '''Validates that torchserve will fail to start in the following scenario:
        1) We use a snapshot file to start torchserve. The snapshot contains reference to "A" model file
        2) The "A" model mar file is accidentally deleted from the model store'''

    # Register model using mgmt api
    snapshot_created_on_management_api_invoke()

    # Now remove the registered model mar file (delete_mar_ fn)
    delete_mar_file_from_model_store(model_store="/workspace/model_store",
                                     model_mar="densenet")

    # Start Torchserve with last generated snapshot file
    snapshot_cfg = glob.glob('logs/config/*snap*.cfg')[0]
    start_torchserve(snapshot_file=snapshot_cfg)
    try:
        response = requests.get('http://127.0.0.1:8081/models/')
    except:
        assert True, "Failed to start Torchserve properly as reqd model mar file is missing!!"
    else:
        assert False, "Something is not right!! Successfully started Torchserve without reqd mar file"
    finally:
        delete_model_store()
        delete_all_snapshots()


def test_replace_mar_file_with_dummy():
    '''Validates that torchserve will fail to start in the following scenario:
        1) We use a snapshot file to start torchserve. The snapshot contains reference to "A" model file
        2) "A" model file gets corrupted or is replaced by some dummy mar file with same name'''

    snapshot_created_on_management_api_invoke()

    # Start Torchserve using last snapshot state
    snapshot_cfg = glob.glob('logs/config/*snap*.cfg')[0]
    start_torchserve(snapshot_file=snapshot_cfg)
    response = requests.get('http://127.0.0.1:8081/models/')
    assert json.loads(response.content)['models'][0]['modelName'] == "densenet161"
    stop_torchserve()

    # Now replace the registered model mar with dummy file
    replace_mar_file_with_dummy_mar_in_model_store(
        model_store="/workspace/model_store", model_mar="densenet161.mar")
    snapshot_cfg = glob.glob('logs/config/*snap*.cfg')[0]
    start_torchserve(snapshot_file=snapshot_cfg)
    try:
        response = requests.get('http://127.0.0.1:8081/models/')
        assert json.loads(response.content)['models'][0]['modelName'] == "densenet161"
    except:
        assert True, "Correct Model mar file not found"
    else:
        assert False, "Something is not right!! Successfully started Torchserve with a dummy mar file"
    finally:
        delete_all_snapshots()
        delete_model_store()


def test_restart_torchserve_with_one_of_model_mar_removed():
    '''Validates that torchserve will fail to start in the following scenario:
        1) We use a snapshot file to start torchserve. The snapshot contains reference to few model files
        2) One of these model mar files are accidentally deleted from the model store'''
    # Register multiple models
    # 1st model
    delete_model_store()
    start_torchserve()
    requests.post(
        'http://127.0.0.1:8081/models?url=https://torchserve.s3.amazonaws.com/mar_files/densenet161.mar')
    time.sleep(15)
    # 2nd model
    requests.post(
        'http://127.0.0.1:8081/models?url=https://torchserve.s3.amazonaws.com/mar_files/mnist.mar')
    time.sleep(15)
    stop_torchserve()

    # Start Torchserve
    start_torchserve()
    response = requests.get('http://127.0.0.1:8081/models/')
    num_of_regd_models = len(json.loads(response.content)['models'])
    stop_torchserve()

    # Now remove the registered model mar file (delete_mar_ fn)
    delete_mar_file_from_model_store(model_store="/workspace/model_store",
                                     model_mar="densenet")

    # Start Torchserve with existing snapshot file containing reference to one of the model mar file
    # which is now missing from the model store
    snapshot_cfg = glob.glob('logs/config/*snap*.cfg')[1]
    start_torchserve(snapshot_file=snapshot_cfg)
    try:
        response = requests.get('http://127.0.0.1:8081/models/')
    except:
        assert True, "Failed to start Torchserve as one of reqd model mar file is missing"
    else:
        assert False, "Something is not right!! Started Torchserve successfully with a " \
                      "reqd model mar file missing from the model store!!"
    finally:
        torchserve_cleanup()