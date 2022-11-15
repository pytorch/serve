import torch

from ts.torch_handler.base_handler import BaseHandler


class ONNXHandler(BaseHandler):
    def __init__(self):
        super().__init__()

    def initialize(self, context):
        super().initialize(context)

    def handle(self, data, context):
        if hasattr(self.model, "run"):
            data = torch.randn(1, 1).to(torch.float32).cpu().numpy()
            # TODO: Should we make this "modelInput configurable", feels complicated
            results = self.model.run(None, {"modelInput": data})

            str_results = ["Prediction Succeeded" for _ in results]
            return str_results
