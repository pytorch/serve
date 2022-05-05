from pyexpat import model
from ts.utils.serve_decorator import serve, torchserve_start, torchserve_stop, archive_model, create_handler, create_torchserve_config
from ts.torch_handler.base_handler import BaseHandler
import os

class ToyHandler(BaseHandler):
    def inference(self):
        return "42"

def test_create_handler():
    create_handler(ToyHandler)
    assert os.path.exists("handler.py")

# TODO: Check if mar file exists as well
def test_archive_model():
    with open("model.pt", "w") as f:
        pass
    archive_model(model_file="model.pt", handler="handler.py")
    assert os.path.exists("model.pt")

def test_create_torchserve_config():
    create_torchserve_config()
    assert os.path.exists("config.properties")

def test_start():
    error = torchserve_start(create_handler(ToyHandler))
    assert error == 0
    error = torchserve_stop()
    assert error == 0

def test_integration():
    """
    Create a handler, config, archive and then serve
    """
    serve(ToyHandler)