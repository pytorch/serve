import torch

from ts.torch_handler.base_handler import BaseHandler


class CompileHandler(BaseHandler):
    def __init__(self):
        super().__init__()

    def initialize(self, context):
        super().initialize(context)

    def preprocess(self, data):
        instances = data[0]["body"]["instances"]
        input_tensor = torch.as_tensor(instances, dtype=torch.float32)
        return input_tensor

    def postprocess(self, data):
        # Convert the output tensor to a list and return
        return data.tolist()[2]
