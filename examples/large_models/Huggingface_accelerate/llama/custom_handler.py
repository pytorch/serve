import logging
from abc import ABC
from typing import Dict

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from ts.context import Context
from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)
logger.info("Transformers version %s", transformers.__version__)


class LlamaHandler(BaseHandler, ABC):
    """
    Transformers handler class for sequence, token classification and question answering.
    """

    def __init__(self):
        super(LlamaHandler, self).__init__()
        self.max_length = None
        self.max_new_tokens = None
        self.tokenizer = None
        self.initialized = False
        self.quant_config = None
        self.return_full_text = True

    def initialize(self, ctx: Context):
        """In this initialize function, the HF large model is loaded and
        partitioned using DeepSpeed.
        Args:
            ctx (context): It is a JSON Object containing information
            pertaining to the model artifacts parameters.
        """
        model_dir = ctx.system_properties.get("model_dir")
        self.max_length = int(ctx.model_yaml_config["handler"]["max_length"])
        self.max_new_tokens = int(ctx.model_yaml_config["handler"]["max_new_tokens"])
        model_name = ctx.model_yaml_config["handler"]["model_name"]
        model_path = f'{model_dir}/{ctx.model_yaml_config["handler"]["model_path"]}'
        self.return_full_text = ctx.model_yaml_config["handler"].get(
            "return_full_text", True
        )
        quantization = ctx.model_yaml_config["handler"].get("quantization", True)
        seed = int(ctx.model_yaml_config["handler"]["manual_seed"])
        torch.manual_seed(seed)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        logger.info("Model %s loaded tokenizer successfully", ctx.model_name)

        if quantization:
            if self.tokenizer.vocab_size >= 128000:
                self.quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
            else:
                self.quant_config = BitsAndBytesConfig(load_in_8bit=True)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="balanced",
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            quantization_config=self.quant_config,
            trust_remote_code=True,
        )
        self.device = next(iter(self.model.parameters())).device
        logger.info("Model %s loaded successfully", ctx.model_name)
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
        input_texts = [self.preprocess_requests(r) for r in requests]

        logger.info("Received texts: '%s'", input_texts)
        inputs = self.tokenizer(
            input_texts,
            max_length=self.max_length,
            padding=True,
            add_special_tokens=True,
            return_tensors="pt",
            truncation=True,
        ).to(self.device)
        return inputs

    def preprocess_requests(self, request: Dict):
        """
        Preprocess request
        Args:
            request (Dict): Request to be decoded.
        Returns:
            str: Decoded input text
        """
        input_text = request.get("data") or request.get("body")
        if isinstance(input_text, (bytes, bytearray)):
            input_text = input_text.decode("utf-8")
        return input_text

    def inference(self, input_batch):
        """
        Predicts the class (or classes) of the received text using the serialized transformers
        checkpoint.
        Args:
            input_batch (tuple): A tuple with two tensors: the batch of input ids and the batch
                                of attention masks, as returned by the preprocess function.
        Returns:
            list: A list of strings with the predicted values for each input text in the batch.
        """
        outputs = self.model.generate(
            **input_batch,
            max_new_tokens=self.max_new_tokens,
        )

        if not self.return_full_text:
            outputs = outputs[:, input_batch["input_ids"].shape[1] :]

        inferences = self.tokenizer.batch_decode(
            outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        logger.info("Generated text: %s", inferences)
        return inferences

    def postprocess(self, inference_output):
        """Post Process Function converts the predicted response into Torchserve readable format.
        Args:
            inference_output (list): It contains the predicted response of the input text.
        Returns:
            (list): Returns a list of the Predictions and Explanations.
        """
        return inference_output
