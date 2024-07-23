import logging
import os
from abc import ABC
from pathlib import Path

import torch
from transformers import LlamaForCausalLM, AutoTokenizer

from ts.torch_handler.base_handler import BaseHandler
from ts.utils.util import check_valid_pt2_backend


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
        self.context = ctx
        self.manifest = ctx.manifest
        properties = ctx.system_properties
        model_dir = properties.get("model_dir")

        model_path = Path(ctx.model_yaml_config["handler"]["model_path"])
        model_name = ctx.model_yaml_config["handler"]["model_name"]
        seed = int(ctx.model_yaml_config["handler"]["manual_seed"])
        torch.manual_seed(seed)

        if "pt2" in ctx.model_yaml_config:
            pt2_value = ctx.model_yaml_config["pt2"]

            if isinstance(pt2_value, str):
                compile_options = dict(backend=pt2_value)
            elif isinstance(pt2_value, dict):
                compile_options = pt2_value
            else:
                raise ValueError("pt2 should be str or dict")

            valid_backend = (
                check_valid_pt2_backend(compile_options["backend"])
                if "backend" in compile_options
                else True
            )
            if not valid_backend:
                raise ValueError("Invalid backend specified in config")

         # Load model weights
        ckpt = os.path.join(model_dir, model_path)

        self.model = LlamaForCausalLM.from_pretrained(ckpt)
        self.model = self.model.eval()
        self.model = torch.compile(self.model, **compile_options)

        self.tokenizer = AutoTokenizer.from_pretrained(ckpt)
        
        logger.info(f"Loaded {model_name} model successfully")

    def preprocess(self, data):
        assert (
            len(data) == 1
        ), "llama-cpp-python is currently only supported with batch_size=1"
        for row in data:
            item = row.get("body")
            return item

    def inference(self, data):
        input_ids = self.tokenizer(data["prompt"], return_tensors="pt").input_ids
        output = self.model.generate(input_ids, max_length=100)
        output_text = self.tokenizer.decode(output[0], skip_special_tokens=True)

        return [output_text]

    def postprocess(self, output):
        return output
