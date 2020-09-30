import requests
import json
import test_utils


def setup_module(module):
    test_utils.torchserve_cleanup()
    response = requests.get("https://torchserve.s3.amazonaws.com/mar_files/mnist.mar", allow_redirects=True)
    open(test_utils.MODEL_STORE + "/mnist.mar", 'wb').write(response.content)


def teardown_module(module):
    test_utils.torchserve_cleanup()


def mnist_model_register_using_non_existent_handler_then_scale_up(synchronous=False):
    """
    Validates that snapshot.cfg is created when management apis are invoked.
    """
    response = requests.post(
        'http://127.0.0.1:8081/models?handler=nehandler&url=mnist.mar')

    # Scale up workers
    if synchronous:
        params = (('min_worker', '2'), ('synchronous', 'True'),)
    else:
        params = (('min_worker', '2'),)

    response = requests.put('http://localhost:8081/models/mnist', params=params)
    # Check if workers got scaled
    response = requests.get('http://127.0.0.1:8081/models/mnist')
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
    # Run inference
    try:
        response = requests.post(url=purl, files=pfiles, timeout=ptimeout)
    except:
        return None
    else:
        return response


def test_mnist_model_register_and_inference_on_valid_model():
    """
    Validates that snapshot.cfg is created when management apis are invoked.
    """
    test_utils.start_torchserve(no_config_snapshots=True)
    # Register mnist model
    test_utils.register_model('mnist', 'mnist.mar')

    files = {
        'data': ('../../examples/image_classifier/mnist/test_data/1.png',
                 open('../../examples/image_classifier/mnist/test_data/1.png', 'rb')),
    }
    response = run_inference_using_url_with_data('http://127.0.0.1:8080/predictions/mnist', files)

    assert (json.loads(response.content)) == 1
    # UnRegister mnist model
    test_utils.unregister_model("mnist")


def test_mnist_model_register_using_non_existent_handler_with_nonzero_workers():
    """
    Validates that a model cannot be registered with a non existent handler if
    the initial number of workers is greater than zero.
    """

    response = requests.post(
        'http://127.0.0.1:8081/models?handler=nehandler&initial_workers=1&url=mnist.mar')
    if json.loads(response.content)['code'] == 500 and \
            json.loads(response.content)['type'] == "InternalServerException":
        assert True, "Internal Server Exception, " \
                     "Cannot register model with non existent handler with non zero workers"
    else:
        assert False, "Something is not right!! Successfully registered model with " \
                      "non existent handler with non zero workers"

    test_utils.unregister_model("mnist")


def test_mnist_model_register_and_scale_using_non_existent_handler():
    """ Bug - Following code block will result in "Buggy" behaviour. If a non-existent handler is used,
    then ideally we should not be able to scale up workers anytime, but currently Torchserve scales up
    background workers. Uncomment it after the Bug is fixed
    """

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

    response = run_inference_using_url_with_data('http://127.0.0.1:8080/predictions/mnist', files)
    if response is None:
        assert True, "Inference failed as the handler is non existent"
    else:
        if json.loads(response.content) == 1:
            assert False, "Something is not right!! Somehow Inference passed " \
                          "despite passing non existent handler"
