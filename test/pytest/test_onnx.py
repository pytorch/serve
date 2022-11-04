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


def test_inference():
    subprocess.run(
        "torch-model-archiver --model-name onnx --version 1.0 --serialized-file linear.onnx --export-path model_store --handler base_handler"
    )
    subprocess.run(
        "torchserve --start --ncs --model-store model_store --models onnx.mar"
    )
    subprocess.run(
        'curl -H "Content-Type: application/json" --data @examples/Huggingface_Transformers/bert_ts.json http://127.0.0.1:8080/explanations/onnx'
    )
