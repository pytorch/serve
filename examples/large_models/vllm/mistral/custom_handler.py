import logging

import torch
import vllm
from vllm import LLM, SamplingParams

from ts.context import Context
from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)
logger.info("vLLM version %s", vllm.__version__)


class CustomHandler(BaseHandler):
    """
    Custom Handler for integrating vLLM
    """

    def __init__(self):
        super().__init__()
        self.max_new_tokens = None
        self.tokenizer = None
        self.initialized = False

    def initialize(self, ctx: Context):
        """In this initialize function, the model is loaded
        Args:
            ctx (context): It is a JSON Object containing information
            pertaining to the model artifacts parameters.
        """
        model_dir = ctx.system_properties.get("model_dir")
        self.max_new_tokens = int(ctx.model_yaml_config["handler"]["max_new_tokens"])
        model_name = ctx.model_yaml_config["handler"]["model_name"]
        model_path = ctx.model_yaml_config["handler"]["model_path"]
        tp_size = ctx.model_yaml_config["torchrun"]["nproc-per-node"]
        seed = int(ctx.model_yaml_config["handler"]["manual_seed"])
        torch.manual_seed(seed)

        self.model = LLM(model=model_path, tensor_parallel_size=tp_size)

        logger.info("Model %s loaded successfully", ctx.model_name)
        self.initialized = True

    def preprocess(self, requests):
        """
        Pre-processing of prompts being sent to TorchServe
        Args:
            requests (list): A list of dictionaries with a "data" or "body" field, each
                            containing the input text to be processed.
        Returns:
            tuple: A tuple with two tensors: the batch of input ids and the batch of
                attention masks.
        """
        input_texts = [data.get("data") or data.get("body") for data in requests]
        input_texts = [
            input_text.decode("utf-8")
            if isinstance(input_text, (bytes, bytearray))
            else input_text
            for input_text in input_texts
        ]
        return input_texts

    def inference(self, input_batch):
        """
        Generates the model response for the given prompt
        Args:
            input_batch : List of input text prompts as returned by the preprocess function.
        Returns:
            list: A list of strings with the predicted values for each input text in the batch.
        """
        logger.info(f"Input text is {input_batch}")
        sampling_params = SamplingParams(max_tokens=self.max_new_tokens)
        outputs = self.model.generate(input_batch, sampling_params=sampling_params)

        logger.info("Generated text: %s", outputs)
        return outputs

    def postprocess(self, inference_output):
        """Post Process Function returns the text response from the vLLM output.
        Args:
            inference_output (list): It contains the response of vLLM
        Returns:
            (list): Returns a list of the Predictions and Explanations.
        """

        return [inf_output.outputs[0].text for inf_output in inference_output]
