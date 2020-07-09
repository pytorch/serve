import subprocess
import time
import os
import glob
import requests
import json

ROOT_DIR = "/workspace/"
MODEL_STORE = ROOT_DIR + "model_store"
# CHANGE THIS TO CORRECT PYTORCH CODE REPOSITORY
CODEBUILD_WD = "/home/deepak/projects/ofc/db_torch/torch_issue_394/serve"


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


def delete_mar_file_from_model_store(model_store=None, model_mar=None):
    model_store = model_store if (model_store != None) else "/workspace/model_store/"
    if model_mar != None:
        for f in glob.glob(model_store + "/" + model_mar + "*"):
            os.remove(f)


def delete_model_store(model_store=None):
    model_store = model_store if (model_store != None) else "/workspace/model_store/"
    for f in glob.glob(model_store + "/*"):
        os.remove(f)


def torchserve_cleanup():
    stop_torchserve()
    delete_model_store()
    delete_all_snapshots()


def run_inference_using_url_with_data(purl=None, pfiles=None, ptimeout=120):
    if purl is None and pfiles is None:
        return
    # Run inference
    try:
        response = requests.post(url=purl, files=pfiles, timeout=ptimeout)
    except:
        assert False, "No response from server"
    else:
        time.sleep(20)
        return response


def test_mnist_model_register_and_inference_on_valid_model():
    ''' 
    Validates that snapshot.cfg is created when management apis are invoked.
    '''
    torchserve_cleanup()
    start_torchserve(no_config_snapshots=True)
    # Register mnist model
    requests.post(
        'http://127.0.0.1:8081/models?initial_workers=1&url=https://torchserve.s3.amazonaws.com/mar_files/mnist.mar')
    time.sleep(20)
    files = {
        'data': ('../../examples/image_classifier/mnist/test_data/1.png',
                 open('../../examples/image_classifier/mnist/test_data/1.png', 'rb')),
    }
    response = run_inference_using_url_with_data('http://127.0.0.1:8080/predictions/mnist', files)
    assert (json.loads(response.content)) == 1
    # UnRegister mnist model
    response = requests.delete('http://localhost:8081/models/mnist')
    time.sleep(10)
    # Clean up
    torchserve_cleanup()


def test_mnist_model_register_using_non_existent_handler_with_nonzero_workers():
    '''
    Validates that a model cannot be registered with a non existent handler if
    the initial number of workers is greater than zero.
    '''
    torchserve_cleanup()
    start_torchserve(no_config_snapshots=True)
    # Register mnist model with some initial workers
    response = requests.post(
        'http://127.0.0.1:8081/models?handler=nehandler&initial_workers=1&url=https://torchserve.s3.amazonaws.com/mar_files/mnist.mar')
    time.sleep(20)
    try:
        if json.loads(response.content)['code'] == 500 and \
                json.loads(response.content)['type'] == "InternalServerException":
            assert False, "Internal Server Exception, " \
                          "Cannot register model with non existent handler with non zero workers"
        else:
            assert True, "Successfully registered model with " \
                         "non existent handler with non zero workers"
    finally:
        torchserve_cleanup()


def mnist_model_register_using_non_existent_handler_then_scale_up():
    '''
    Validates that snapshot.cfg is created when management apis are invoked.
    '''
    torchserve_cleanup()
    start_torchserve(no_config_snapshots=True)
    # Register mnist model with default(0) workers
    response = requests.post(
        'http://127.0.0.1:8081/models?handler=nehandler&url=https://torchserve.s3.amazonaws.com/mar_files/mnist.mar')
    time.sleep(20)

    # Scale up workers
    params = (('min_worker', '2'),)
    response = requests.put('http://localhost:8081/models/mnist', params=params)
    time.sleep(20)
    # Check if workers got scaled
    response = requests.get('http://127.0.0.1:8081/models/mnist')
    return response


def test_mnist_model_register_and_scale_using_non_existent_handler():
    # Register & Scale model
    response = mnist_model_register_using_non_existent_handler_then_scale_up()
    mnist_list = json.loads(response.content)
    assert len(mnist_list[0]['workers']) > 1
    # Cleanup

    # UnRegister mnist model
    response = requests.delete('http://localhost:8081/models/mnist')
    time.sleep(10)

    torchserve_cleanup()


def test_mnist_model_register_scale_inference_with_non_existent_handler():
    response = mnist_model_register_using_non_existent_handler_then_scale_up()
    mnist_list = json.loads(response.content)
    assert len(mnist_list[0]['workers']) > 1
    files = {
        'data': ('../../examples/image_classifier/mnist/test_data/1.png',
                 open('../../examples/image_classifier/mnist/test_data/1.png', 'rb')),
    }
    try:
        response = run_inference_using_url_with_data('http://127.0.0.1:8080/predictions/mnist', files)
        assert (json.loads(response.content)) == 1
        # UnRegister mnist model
        response = requests.delete('http://localhost:8081/models/mnist')
        time.sleep(10)
    finally:
        # Remove the non-existent/invalid handler based model mar file from model-store
        delete_mar_file_from_model_store("/workspace/model_store/", "mnist")
        # Cleanup
        torchserve_cleanup()
