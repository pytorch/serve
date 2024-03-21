import logging
import os
from abc import ABC

import torch
from llama_cpp import Llama

from ts.protocol.otf_message_handler import send_intermediate_predict_response
from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)


class LlamaCppHandler(BaseHandler, ABC):
    def __init__(self):
        super(LlamaCppHandler, self).__init__()
        self.initialized = False

    def initialize(self, ctx):
        """In this initialize function, the HF large model is loaded and
        partitioned using DeepSpeed.
        Args:
            ctx (context): It is a JSON Object containing information
            pertaining to the model artifacts parameters.
        """
        model_path = os.environ["LLAMA2_Q4_MODEL"]
        model_name = ctx.model_yaml_config["handler"]["model_name"]
        seed = int(ctx.model_yaml_config["handler"]["manual_seed"])
        torch.manual_seed(seed)

        self.model = Llama(model_path=model_path)
        logger.info(f"Loaded {model_name} model successfully")

    def preprocess(self, data):
        assert (
            len(data) == 1
        ), "llama-cpp-python is currently only supported with batch_size=1"
        for row in data:
            item = row.get("body")
            return item

    def inference(self, data):
        params = data["params"]
        tokens = self.model.tokenize(bytes(data["prompt"], "utf-8"))
        generation_kwargs = dict(
            tokens=tokens,
            temp=params["temperature"],
            top_p=params["top_p"],
        )
        count = 0
        for token in self.model.generate(**generation_kwargs):
            if count >= params["max_new_tokens"]:
                break

            count += 1
            new_text = self.model.detokenize([token])
            send_intermediate_predict_response(
                [new_text],
                self.context.request_ids,
                "Intermediate Prediction success",
                200,
                self.context,
            )
        return [""]

    def postprocess(self, output):
        return output
