import logging
from abc import ABC

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from ts.context import Context
from ts.handler_utils.distributed.deepspeed import get_ds_engine
from ts.torch_handler.distributed.base_deepspeed_handler import BaseDeepSpeedHandler

logger = logging.getLogger(__name__)
logger.info("Transformers version %s", transformers.__version__)


class TransformersSeqClassifierHandler(BaseDeepSpeedHandler, ABC):
    """
    Transformers handler class for sequence, token classification and question answering.
    """

    def __init__(self):
        super(TransformersSeqClassifierHandler, self).__init__()
        self.max_length = None
        self.max_new_tokens = None
        self.tokenizer = None
        self.initialized = False

    def initialize(self, ctx: Context):
        """In this initialize function, the HF large model is loaded and
        partitioned using DeepSpeed.
        Args:
            ctx (context): It is a JSON Object containing information
            pertaining to the model artefacts parameters.
        """
        super().initialize(ctx)
        model_dir = ctx.system_properties.get("model_dir")
        self.max_length = int(ctx.model_yaml_config["handler"]["max_length"])
        self.max_new_tokens = int(ctx.model_yaml_config["handler"]["max_new_tokens"])
        seed = int(ctx.model_yaml_config["handler"]["manual_seed"])
        torch.manual_seed(seed)

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, padding_side="left")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir, torch_dtype=torch.float16
        )
        self.model.eval()

        ds_engine = get_ds_engine(self.model, ctx)
        self.model = ds_engine.module
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
        input_texts = [data.get("data") or data.get("body") for data in requests]
        input_ids_batch, attention_mask_batch = [], []
        for input_text in input_texts:
            input_ids, attention_mask = self.encode_input_text(input_text)
            input_ids_batch.append(input_ids)
            attention_mask_batch.append(attention_mask)
        input_ids_batch = torch.cat(input_ids_batch, dim=0).to(self.device)
        attention_mask_batch = torch.cat(attention_mask_batch, dim=0).to(self.device)
        return input_ids_batch, attention_mask_batch

    def encode_input_text(self, input_text):
        """
        Encodes a single input text using the tokenizer.
        Args:
            input_text (str): The input text to be encoded.
        Returns:
            tuple: A tuple with two tensors: the encoded input ids and the attention mask.
        """
        if isinstance(input_text, (bytes, bytearray)):
            input_text = input_text.decode("utf-8")
        logger.info("Received text: '%s'", input_text)
        inputs = self.tokenizer.encode_plus(
            input_text,
            max_length=self.max_length,
            padding=True,
            add_special_tokens=True,
            return_tensors="pt",
            truncation=True,
        )
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        return input_ids, attention_mask

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
        input_ids_batch, attention_mask_batch = input_batch
        input_ids_batch = input_ids_batch.to(self.device)
        outputs = self.model.generate(
            input_ids_batch,
            attention_mask=attention_mask_batch,
            max_length=self.max_new_tokens,
        )

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
