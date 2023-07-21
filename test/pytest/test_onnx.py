import subprocess

import torch
import torch.onnx


class ToyModel(torch.nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.linear1 = torch.nn.Linear(1, 1)
        self.linear2 = torch.nn.Linear(1, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x


# For a custom model you still need to manually author your converter, as far as I can tell there isn't a nice out of the box that exists
def test_convert_to_onnx():
    model = ToyModel()
    dummy_input = torch.randn(1, 1)
    model_path = "linear.onnx"
    # set the model to inference mode
    model.eval()

    # Let's create a dummy input tensor

    # Export the model
    torch.onnx.export(
        model,  # model being run
        dummy_input,  # model input (or a tuple for multiple inputs)
        model_path,  # where to save the model
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=10,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=["modelInput"],  # the model's input names
        output_names=["modelOutput"],  # the model's output names
        dynamic_axes={
            "modelInput": {0: "batch_size"},  # variable length axes
            "modelOutput": {0: "batch_size"},
        },
    )


def test_model_packaging_and_start():
    subprocess.run("mkdir model_store", shell=True)
    subprocess.run(
        "torch-model-archiver -f --model-name onnx --version 1.0 --serialized-file linear.onnx --export-path model_store --handler onnx_handler.py",
        shell=True,
        check=True,
    )


def test_model_start():
    subprocess.run(
        "torchserve --start --ncs --model-store model_store --models onnx.mar",
        shell=True,
        check=True,
    )


def test_inference():
    subprocess.run(
        "curl -X POST http://127.0.0.1:8080/predictions/onnx --data-binary '1'",
        shell=True,
    )


def test_stop():
    subprocess.run("torchserve --stop", shell=True, check=True)
