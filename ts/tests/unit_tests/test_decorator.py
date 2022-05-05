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
    assert os.path.exists("model.pt")
    archive_model(model_file="model.pt", handler="handler.py")
    assert os.path.exists(os.path.join("model_store", "model.mar"))

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
    output =os.system("curl http://127.0.0.1:8080/predictions/model")
    assert output