import os
import logging
from abc import ABC

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from ts.context import Context
from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)
logger.info("Transformers version %s", transformers.__version__)

IPEX_ENABLE = False
if os.environ.get("TS_IPEX_ENABLE", "false") == "true":
    try:
        import intel_extension_for_pytorch as ipex
        try:
            ipex._C.disable_jit_linear_repack()
        except Exception:
            pass
        IPEX_ENABLE = True
    except ImportError as error:
        logger.warning(
            "IPEX is enabled but intel-extension-for-pytorch is not installed. Proceeding without IPEX."
        )
        IPEX_ENABLE = False

class CodeGenHandler(BaseHandler, ABC):

    def __init__(self):
        super(CodeGenHandler, self).__init__()

    def initialize(self, ctx: Context):
        self.max_length = int(ctx.model_yaml_config["handler"]["max_length"])
        model_name = ctx.model_yaml_config["handler"]["model_name"]
        
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model = self.model.eval().to("cpu")
        
        logger.info("Model %s loading tokenizer", ctx.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if IPEX_ENABLE:
            logger.info("Model %s optimzied with Intel® Extension for PyTorch*", ctx.model_name)
            self.model = self.model.to(memory_format=torch.channels_last)
            self.model = ipex._optimize_transformers(self.model, inplace=True)
            
            
        logger.info("Model %s loaded successfully", ctx.model_name)

    def preprocess(self, requests):
        input_ids_batch = None
        for idx, data in enumerate(requests):
            input_text = data.get("data")
            if input_text is None:
                input_text = data.get("body")
            if isinstance(input_text, (bytes, bytearray)):
                input_text = input_text.decode("utf-8")
            logger.info("Received text: '%s'", input_text)
            
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids
        return input_ids 
        
    def inference(self, input_batch):
        inferences = []
        outputs = self.model.generate(input_batch, max_length=self.max_length)
        for i, x in enumerate(outputs):
            inferences.append(
                self.tokenizer.decode(outputs[i], skip_special_tokens=True)
            )
        return inferences

    def postprocess(self, inference_output):
        return inference_output