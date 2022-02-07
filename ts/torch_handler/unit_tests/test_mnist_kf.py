# pylint: disable=W0621
# Using the same name as global function is part of pytest
"""
Basic unit test for ImageClassifier class.
Ensures it can load and execute an example model
"""

import sys
import io
import pytest
from PIL import Image
import torchvision.transforms as transforms
from ts.torch_handler.request_envelope.kserve import KServeEnvelope
from examples.image_classifier.mnist.mnist_handler import MNISTDigitClassifier as MNISTClassifier
from .test_utils.mock_context import MockContext

sys.path.append('ts/torch_handler/unit_tests/models/tmp')


image_processing = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ])

@pytest.fixture()
def model_setup():
    context = MockContext(model_pt_file="mnist_cnn.pt", model_file="mnist.py")
    with open("ts/torch_handler/unit_tests/models/tmp/test_data/0.png", "rb") as fin:
        image_bytes = fin.read()
    return (context, image_bytes)

def test_initialize(model_setup):
    model_context, _ = model_setup
    handler = MNISTClassifier()
    handler.initialize(model_context)
    assert(True)
    return handler

def test_handle(model_setup):
    context, bytes_array = model_setup
    handler = test_initialize(model_setup)
    test_data = [{'data': bytes_array}]
    #testing for predict API
    results = handler.handle(test_data, context)
    assert(results[0] in range(0, 9))

def test_initialize_kf(model_setup):
    model_context, _ = model_setup
    handler = MNISTClassifier()
    handler.initialize(model_context)
    envelope = KServeEnvelope(handler.handle)
    assert(True)
    return envelope

def test_handle_kf(model_setup):
    context, bytes_array = model_setup
    image = Image.open(io.BytesIO(bytes_array))
    image_list = image_processing(image).tolist()
    envelope = test_initialize_kf(model_setup)
    test_data = {'body': {'instances': [{'data': image_list}]}}

    #testing for predict API
    results = envelope.handle([test_data], context)
    assert(results[0]["predictions"][0] in range(0, 9))
