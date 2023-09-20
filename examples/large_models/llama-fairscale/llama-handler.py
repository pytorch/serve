import logging
import time
from abc import ABC

import packaging.version
import requests
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from ts.torch_handler.base_handler import BaseHandler
from llama import Llama

logger = logging.getLogger(__name__)
logger.info("Transformers version %s", transformers.__version__)
if packaging.version.parse(torch.__version__) >= packaging.version.parse("2.0.0"):
    logger.info("PyTorch version is 2.0.0 or greater")
else:
    logger.info(
        "PyTorch version is less than 2.0.0, initializing with meta device needs PyTorch 2.0.0 and greater"
    )


class LlamaFairscaleHandler(BaseHandler,ABC):
    """
    Transformers handler class for sequence, token classification and question answering.
    """

    def __init__(self):
        super(LlamaFairscaleHandler, self).__init__()
        self.initialized = False

    def initialize(self, ctx):
        """
        In this initialize function, the llama model is loaded using Fairscale and
        partitioned into multiple stages each on one device using PiPPy.
        Args:
            ctx (context): It is a JSON Object containing information
            pertaining to the model artefacts parameters.
        """
        # super().initialize(ctx)
        self.manifest = ctx.manifest
        properties = ctx.system_properties
        model_dir = properties.get("model_dir")
        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id"))
            if torch.cuda.is_available() and properties.get("gpu_id") is not None
            else "cpu"
        )

        model_path = ctx.model_yaml_config["handler"]["model_path"]
        tokenizer_path = ctx.model_yaml_config["handler"]["tokenizer_path"]
        max_seq_len = ctx.model_yaml_config["handler"]["max_seq_len"]
        max_batch_size = ctx.model_yaml_config["handler"]["max_batch_size"]
        seed = ctx.model_yaml_config["handler"]["manual_seed"]
        dtype_str = ctx.model_yaml_config["handler"]["dtype"]
        self.max_new_tokens = ctx.model_yaml_config["handler"]["max_new_tokens"]
        self.temperature = ctx.model_yaml_config["handler"]["temperature"]
        self.top_p = ctx.model_yaml_config["handler"]["top_p"]
        
        torch.manual_seed(seed)

        dtypes = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}

        dtype = dtypes.get(dtype_str, torch.float32)
        if dtype != torch.float32 and dtype_str not in dtypes:
            logger.info(
                f"Unsupported data type {dtype_str}, "
                "please submit a PR to support it. Falling back to fp32 now."
            )

        logger.info("Instantiating Llama model")
        self.model = Llama.build(
        ckpt_dir=model_path,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        )

        logger.info("Llama model from path %s loaded successfully", model_dir)

        self.initialized = True

    def preprocess(self, requests):
        """
        Basic text preprocessing, based on the user's choice of application mode.
        Args:
            requests (list): A list of dictionaries with a "data" or "body" field, each
                            containing the input text to be processed.
        Returns:
            tuple: A tuple with two tensors: the batch of input ids and the batch of
                attention masks.
        """
        input_texts = [data.get("data") or data.get("body") for data in requests]
        input_ids_batch = []
        for input_text in input_texts:
            input_ids = self.prep_input_text(input_text)
            input_ids_batch.append(input_ids)
        # input_ids_batch = torch.cat(input_ids_batch, dim=0)
        return input_ids_batch

    def prep_input_text(self, input_text):
        """
        preparing a single input text using the tokenizer.
        Args:
            input_text (str): The input text to be encoded.
        Returns:
            decoded input text
        """
        if isinstance(input_text, (bytes, bytearray)):
            input_text = input_text.decode("utf-8")
        logger.info("Received text: '%s'", input_text)
        
        return input_text

    def inference(self, input_batch):
        """
        Predicts the class (or classes) of the received text using the serialized transformers
        checkpoint.
        Args:
            input_batch : a batch of input texts
        Returns:
            list: A list of strings with the predicted values for each input text in the batch.
        """
        input_ids_batch = input_batch
        input_ids_batch = input_ids_batch
        results = self.model.text_completion(
                input_batch,
                max_gen_len=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
            )
        
        
        return results

    def postprocess(self, inference_output):
        """Post Process Function converts the predicted response into Torchserve readable format.
        Args:
            inference_output (list): It contains the predicted response of the input text.
        Returns:
            (list): Returns a list of the Predictions and Explanations.
        """
        
        logger.info("Generated text: %s", inference_output)
        
        return inference_output
