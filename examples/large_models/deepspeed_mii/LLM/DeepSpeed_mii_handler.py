import logging
import os
from abc import ABC

import mii

from ts.context import Context
from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)
logger.info("DeepSpeed MII version %s", mii.__version__)


class DeepSpeedMIIHandler(BaseHandler, ABC):
    """
    Diffusers handler class for text to image generation.
    """

    def __init__(self):
        self.device = int(os.getenv("LOCAL_RANK", 0))
        self.initialized = False

    def initialize(self, ctx: Context):
        """In this initialize function, the Stable Diffusion model is loaded and
        initialized here.
        Args:
            ctx (context): It is a JSON Object containing information
            pertaining to the model artifacts parameters.
        """
        model_dir = ctx.system_properties.get("model_dir")
        model_name = ctx.model_yaml_config["handler"]["model_name"]
        model_path = ctx.model_yaml_config["handler"]["model_path"]
        self.max_new_tokens = int(ctx.model_yaml_config["handler"]["max_new_tokens"])

        model_config = {
            "tensor_parallel": int(ctx.model_yaml_config["handler"]["tensor_parallel"]),
            "max_length": int(ctx.model_yaml_config["handler"]["max_length"]),
        }
        self.pipe = mii.pipeline(
            model_name_or_path=model_path,
            model_config=model_config,
        )
        logger.info("Model %s loaded successfully", model_name)
        self.initialized = True

    def preprocess(self, requests):
        """Basic text preprocessing, of the user's prompt.
        Args:
            requests (str): The Input data in the form of text is passed on to the preprocess
            function.
        Returns:
            list : The preprocess function returns a list of prompts.
        """
        inputs = []
        for _, data in enumerate(requests):
            input_text = data.get("data")
            if input_text is None:
                input_text = data.get("body")
            if isinstance(input_text, (bytes, bytearray)):
                input_text = input_text.decode("utf-8")
            logger.info("Received text: '%s'", input_text)
            inputs.append(input_text)
        return inputs

    def inference(self, inputs):
        """Generates the image relevant to the received text.
        Args:
            input_batch (list): List of Text from the pre-process function is passed here
        Returns:
            list : It returns a list of the generate images for the input text
        """
        inferences = self.pipe(
            inputs, max_new_tokens=self.max_new_tokens
        ).generated_texts

        logger.info("Generated text: %s", inferences)
        return inferences

    def postprocess(self, inference_output):
        """Post Process Function converts the generated image into Torchserve readable format.
        Args:
            inference_output (list): It contains the generated image of the input text.
        Returns:
            (list): Returns a list of the images.
        """

        return inference_output
