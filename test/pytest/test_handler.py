import os
import requests
import json
import test_utils
import numpy as np

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../")
snapshot_file_kf = os.path.join(REPO_ROOT,"test/config_kf.properties")
snapshot_file_tf = os.path.join(REPO_ROOT,"test/config_ts.properties")

def getAPIS(snapshot_file):
    MANAGEMENT_API = "http://127.0.0.1:8081"
    INFERENCE_API = "http://127.0.0.1:8080"
    
    with open(snapshot_file, "r") as fp:
        lines = fp.readlines()
    for line in lines:
        if "management_address" in line:
            MANAGEMENT_API = line.split("=")[1]
        if "inference_address" in line:
            INFERENCE_API = line.split("=")[1]
    
    return (MANAGEMENT_API, INFERENCE_API)


KF_MANAGEMENT_API, KF_INFERENCE_API = getAPIS(snapshot_file_kf)
TF_MANAGEMENT_API, TF_INFERENCE_API = getAPIS(snapshot_file_tf)

def setup_module(module):
    test_utils.torchserve_cleanup()
    response = requests.get("https://torchserve.pytorch.org/mar_files/mnist.mar", allow_redirects=True)
    open(test_utils.MODEL_STORE + "/mnist.mar", 'wb').write(response.content)


def teardown_module(module):
    test_utils.torchserve_cleanup()


def mnist_model_register_using_non_existent_handler_then_scale_up(synchronous=False):
    """
    Validates that snapshot.cfg is created when management apis are invoked.
    """
    response = requests.post(
        TF_MANAGEMENT_API + '/models?handler=nehandler&url=mnist.mar')

    # Scale up workers
    if synchronous:
        params = (('min_worker', '2'), ('synchronous', 'True'),)
    else:
        params = (('min_worker', '2'),)

    response = requests.put(TF_MANAGEMENT_API + '/models/mnist', params=params)
    # Check if workers got scaled
    response = requests.get(TF_MANAGEMENT_API + '/models/mnist')
    return response


def mnist_model_register_and_scale_using_non_existent_handler_synchronous():
    # Register & Scale model
    response = mnist_model_register_using_non_existent_handler_then_scale_up(synchronous=True)
    mnist_list = json.loads(response.content)
    try:
        # Workers should not scale up
        assert len(mnist_list[0]['workers']) == 0
    finally:
        # UnRegister mnist model
        test_utils.unregister_model("mnist")


def mnist_model_register_and_scale_using_non_existent_handler_asynchronous():
    # Register & Scale model
    response = mnist_model_register_using_non_existent_handler_then_scale_up()
    mnist_list = json.loads(response.content)
    try:
        # Workers should not scale up
        assert len(mnist_list[0]['workers']) == 0
    finally:
        # UnRegister mnist model
        test_utils.unregister_model("mnist")


def run_inference_using_url_with_data(purl=None, pfiles=None, ptimeout=120):
    if purl is None and pfiles is None:
        return
    try:
        response = requests.post(url=purl, files=pfiles, timeout=ptimeout)
    except:
        return None
    else:
        return response

def run_inference_using_url_with_data_json(purl=None, json_input=None, ptimeout=120):
    if purl is None and pfiles is None:
        return
    try:
        response = requests.post(url=purl, json=json_input, timeout=ptimeout)
    except:
        return None
    else:
        return response


def test_mnist_model_register_and_inference_on_valid_model():
    """
    Validates that snapshot.cfg is created when management apis are invoked.
    """
    test_utils.start_torchserve(no_config_snapshots=True)
    test_utils.register_model('mnist', 'mnist.mar')

    files = {
        'data': ('../../examples/image_classifier/mnist/test_data/1.png',
                 open('../../examples/image_classifier/mnist/test_data/1.png', 'rb')),
    }
    response = run_inference_using_url_with_data(TF_INFERENCE_API + '/predictions/mnist', files)

    assert (json.loads(response.content)) == 1
    test_utils.unregister_model("mnist")


def test_mnist_model_register_using_non_existent_handler_with_nonzero_workers():
    """
    Validates that a model cannot be registered with a non existent handler if
    the initial number of workers is greater than zero.
    """

    response = requests.post(
        TF_MANAGEMENT_API + '/models?handler=nehandlermodels&initial_workers=1&url=mnist.mar')
    if json.loads(response.content)['code'] == 500 and \
            json.loads(response.content)['type'] == "InternalServerException":
        assert True, "Internal Server Exception, " \
                     "Cannot register model with non existent handler with non zero workers"
    else:
        assert False, "Something is not right!! Successfully registered model with " \
                      "non existent handler with non zero workers"

    test_utils.unregister_model("mnist")


def test_mnist_model_register_scale_inference_with_non_existent_handler():
    response = mnist_model_register_using_non_existent_handler_then_scale_up()
    mnist_list = json.loads(response.content)
    assert len(mnist_list[0]['workers']) > 1
    files = {
        'data': ('../../examples/image_classifier/mnist/test_data/1.png',
                 open('../../examples/image_classifier/mnist/test_data/1.png', 'rb')),
    }

    response = run_inference_using_url_with_data(TF_INFERENCE_API + '/predictions/mnist', files)
    if response is None:
        assert True, "Inference failed as the handler is non existent"
    else:
        if json.loads(response.content) == 1:
            assert False, "Something is not right!! Somehow Inference passed " \
                          "despite passing non existent handler"


def test_mnist_model_register_and_inference_on_valid_model_explain():
    """
    Validates that snapshot.cfg is created when management apis are invoked.
    """
    test_utils.start_torchserve(no_config_snapshots=True)
    test_utils.register_model('mnist', 'mnist.mar')

    files = {
        'data': ('../../examples/image_classifier/mnist/test_data/1.png',
                 open('../../examples/image_classifier/mnist/test_data/1.png', 'rb')),
    }
    response = run_inference_using_url_with_data(TF_INFERENCE_API + '/explanations/mnist', files)

    assert np.array(json.loads(response.content)).shape == (1, 28, 28)
    test_utils.unregister_model("mnist")


def test_kfserving_mnist_model_register_and_inference_on_valid_model():
    """
    Validates that snapshot.cfg is created when management apis are invoked for kfserving.
    """
    test_utils.start_torchserve(snapshot_file = snapshot_file_kf)
    test_utils.register_model('mnist', 'mnist.mar')

    input_json = "../../kubernetes/kfserving/kf_request_json/mnist.json"
    with open(input_json, 'r') as f:
        s = f.read()
        s = s.replace('\'','\"')
        data = json.loads(s)

    response = run_inference_using_url_with_data_json(KF_INFERENCE_API + '/v1/models/mnist:predict', data)

    assert (json.loads(response.content)['predictions'][0]) == 2
    test_utils.unregister_model("mnist")


def test_kfserving_mnist_model_register_scale_inference_with_non_existent_handler():
    response = mnist_model_register_using_non_existent_handler_then_scale_up()
    mnist_list = json.loads(response.content)
    assert len(mnist_list[0]['workers']) > 1
    
    input_json = "../../kubernetes/kfserving/kf_request_json/mnist.json"
    with open(input_json, 'r') as f:
        s = f.read()
        s = s.replace('\'','\"')
        data = json.loads(s)

    response = run_inference_using_url_with_data_json(KF_INFERENCE_API + '/v1/models/mnist:predict', data)
    
    if response is None:
        assert True, "Inference failed as the handler is non existent"
    else:
        if json.loads(response.content) == 1:
            assert False, "Something is not right!! Somehow Inference passed " \
                          "despite passing non existent handler"


def test_kfserving_mnist_model_register_and_inference_on_valid_model_explain():
    """
    Validates the kfserving model explanations.
    """
    test_utils.start_torchserve(snapshot_file = snapshot_file_kf)
    test_utils.register_model('mnist', 'mnist.mar')

    input_json = "../../kubernetes/kfserving/kf_request_json/mnist.json"
    with open(input_json, 'r') as f:
        s = f.read()
        s = s.replace('\'','\"')
        data = json.loads(s)

    response = run_inference_using_url_with_data_json(KF_INFERENCE_API + '/v1/models/mnist:explain', data)

    assert np.array(json.loads(response.content)['explanations']).shape == (1, 1, 28, 28)
    test_utils.unregister_model("mnist")