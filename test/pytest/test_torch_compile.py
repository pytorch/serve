import subprocess

import pytest
import torch

from ts.torch_handler.BaseHandler import check_pt2_enabled

CURR_FILE_PATH = Path(__file__).parent


class ToyModel(torch.nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.linear1 = torch.nn.Linear(1, 1)
        self.linear2 = torch.nn.Linear(1, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x


@pytest.mark.skipif(not check_pt2_enabled(), reason="PyTorch version is older than 2.0")
def test_serialize_model():
    model = ToyModel()
    torch.save(model, "pt2.pt")


@pytest.mark.skipif(not check_pt2_enabled(), reason="PyTorch version is older than 2.0")
def test_model_packaging_and_start():
    config_path = os.path.join(CURR_FILE_PATH, "yaml_config", "pt2.yaml")
    subprocess.run("mkdir model_store", shell=True)
    subprocess.run(
        f"torch-model-archiver -f --model-name pt2 --version 1.0 --serialized-file pt2.pt --export-path model_store --handler base_handler --config {config_path}",
        shell=True,
        check=True,
    )


@pytest.mark.skipif(not check_pt2_enabled(), reason="PyTorch version is older than 2.0")
def test_model_start():
    subprocess.run(
        "torchserve --start --ncs --model-store model_store --models pt2.mar",
        shell=True,
        check=True,
    )


@pytest.mark.skipif(
    not check_pt2_enabled(), reason="PyTorch version is older than PT 2.0"
)
def test_inference_and_compilation():
    subprocess.run(
        "curl -X POST http://127.0.0.1:8080/predictions/pt2 --data-binary '1'",
        shell=True,
    )


@pytest.mark.skipif(
    not check_pt2_enabled(), reason="PyTorch version is older than PT 2.0"
)
def test_stop():
    subprocess.run("torchserve --stop", shell=True, check=True)
