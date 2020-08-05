import subprocess
import time
import os
import glob
import requests
import json
from os import path

ROOT_DIR = "/workspace/"
MODEL_STORE = ROOT_DIR + "model_store/"
# CHANGE THIS TO CORRECT PYTORCH CODE REPOSITORY
CODEBUILD_WD = path.abspath(path.join(__file__, "../../.."))


def start_torchserve(model_store=None, snapshot_file=None, no_config_snapshots=False):
    stop_torchserve()
    cmd = ["torchserve", "--start"]
    model_store = model_store if (model_store != None) else MODEL_STORE
    cmd.extend(["--model-store", MODEL_STORE])
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


def run_inference_using_url_with_data(purl=None, pfiles=None, ptimeout=120):
    if purl is None and pfiles is None:
        return
    # Run inference
    try:
        response = requests.post(url=purl, files=pfiles, timeout=ptimeout)
    except:
        return None
    else:
        time.sleep(20)
        return response


def test_mnist_model_register_and_inference_on_valid_model():
    ''' 
    Validates that snapshot.cfg is created when management apis are invoked.
    '''
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


def test_mnist_model_register_using_non_existent_handler_with_nonzero_workers():
    '''
    Validates that a model cannot be registered with a non existent handler if
    the initial number of workers is greater than zero.
    '''
    response = requests.post(
        'http://127.0.0.1:8081/models?handler=nehandler&initial_workers=1&url=https://torchserve.s3.amazonaws.com/mar_files/mnist.mar')
    time.sleep(20)
    if json.loads(response.content)['code'] == 500 and \
            json.loads(response.content)['type'] == "InternalServerException":
        assert True, "Internal Server Exception, " \
                      "Cannot register model with non existent handler with non zero workers"
    else:
        assert False, "Something is not right!! Successfully registered model with " \
                      "non existent handler with non zero workers"


def mnist_model_register_using_non_existent_handler_then_scale_up(synchronous=False):
    '''
    Validates that snapshot.cfg is created when management apis are invoked.
    '''
    response = requests.post(
        'http://127.0.0.1:8081/models?handler=nehandler&url=https://torchserve.s3.amazonaws.com/mar_files/mnist.mar')
    time.sleep(20)

    # Scale up workers
    if synchronous is False:
        params = (('min_worker', '2'),)
    else:
        params = (('min_worker', '2'),('synchronous', 'True'),)
    response = requests.put('http://localhost:8081/models/mnist', params=params)
    time.sleep(20)
    # Check if workers got scaled
    response = requests.get('http://127.0.0.1:8081/models/mnist')
    return response


def mnist_model_register_and_scale_using_non_existent_handler_asynchronous():
    # Register & Scale model
    response = mnist_model_register_using_non_existent_handler_then_scale_up()
    mnist_list = json.loads(response.content)
    try:
        #Workers should not scale up
        assert len(mnist_list[0]['workers']) == 0
    finally:
        # UnRegister mnist model
        response = requests.delete('http://localhost:8081/models/mnist')
        time.sleep(10)


def mnist_model_register_and_scale_using_non_existent_handler_synchronous():
    # Register & Scale model
    response = mnist_model_register_using_non_existent_handler_then_scale_up(synchronous=True)
    mnist_list = json.loads(response.content)
    try:
        #Workers should not scale up
        assert len(mnist_list[0]['workers']) == 0
    finally:
        # UnRegister mnist model
        response = requests.delete('http://localhost:8081/models/mnist')
        time.sleep(10)

def test_mnist_model_register_and_scale_using_non_existent_handler():
    ''' Bug - Following code block will result in "Buggy" behaviour. If a non-existent handler is used,
    then ideally we should not be able to scale up workers anytime, but currently Torchserve scales up
    background workers. Uncomment it after the Bug is fixed
    '''

    # mnist_model_register_and_scale_using_non_existent_handler_synchronous()
    # mnist_model_register_and_scale_using_non_existent_handler_asynchronous()


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
        if response is None:
            assert True, "Inference failed as the handler is non existent"
        else:
            if json.loads(response.content) == 1:
                assert False, "Something is not right!! Somehow Inference passed " \
                              "despite passing non existent handler"
    finally:
        # Remove the non-existent/invalid handler based model mar file from model-store
        delete_mar_file_from_model_store("/workspace/model_store/", "mnist")
        # Final Cleanup
        torchserve_cleanup()