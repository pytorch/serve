# resnet50_handler.py
#

import io
import os

import PIL.Image as Image
import rebel  # RBLN Runtime
import torch
from torchvision.models import ResNet50_Weights

from ts.torch_handler.base_handler import BaseHandler


class Resnet50Handler(BaseHandler):
    def __init__(self):
        self._context = None
        self.initialized = False
        self.model = None
        self.weights = None

    def initialize(self, context):
        """
        Initialize model. This will be called during model loading time
        :param context: Initial context contains model server system properties.
        :return:
        """
        self._context = context
        #  load the model, refer 'custom handler class' above for details
        model_dir = context.system_properties.get("model_dir")
        serialized_file = context.manifest["model"].get("serializedFile")
        model_path = os.path.join(model_dir, serialized_file)
        if not os.path.isfile(model_path):
            raise RuntimeError(
                f"[RBLN ERROR] File not found at the specified model_path({model_path})."
            )

        self.module = rebel.Runtime(model_path, tensor_type="pt")
        self.weights = ResNet50_Weights.DEFAULT
        self.initialized = True

    def preprocess(self, data):
        """
        Transform raw input into model input data.
        :param batch: list of raw requests, should match batch size
        :return: list of preprocessed model input data
        """
        input_data = data[0].get("data")
        if input_data is None:
            input_data = data[0].get("body")
        assert input_data is not None, print(
            "[RBLN][ERROR] Data not found with client request."
        )
        if not isinstance(input_data, (bytes, bytearray)):
            raise ValueError("[RBLN][ERROR] Preprocessed data is not binary data.")

        try:
            image = Image.open(io.BytesIO(input_data))
        except Exception as e:
            raise ValueError(f"[RBLN][ERROR]Invalid image data: {e}")
        prep = self.weights.transforms()
        batch = prep(image).unsqueeze(0)
        preprocessed_data = batch.numpy()

        return torch.from_numpy(preprocessed_data)

    def inference(self, model_input):
        """
        Internal inference methods
        :param model_input: transformed model input data
        :return: list of inference output in NDArray
        """

        model_output = self.module.run(model_input)

        return model_output

    def postprocess(self, inference_output):
        """
        Return inference result.
        :param inference_output: list of inference output
        :return: list of predict results
        """
        score, class_id = torch.topk(inference_output, 1, dim=1)
        category_name = self.weights.meta["categories"][class_id]
        return category_name

    def handle(self, data, context):
        """
        Invoke by TorchServe for prediction request.
        Do pre-processing of data, prediction using model and postprocessing of prediciton output
        :param data: Input data for prediction
        :param context: Initial context contains model server system properties.
        :return: prediction output
        """
        model_input = self.preprocess(data)
        model_output = self.inference(model_input)
        category_name = self.postprocess(model_output)

        print("[RBLN][INFO] Top1 category: ", category_name)

        return [{"result": category_name}]
