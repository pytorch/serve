import json
import logging
import os
import zipfile
from abc import ABC

import mii
import numpy as np
import torch

from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)
logger.info("DeepSpeed MII version %s", mii.__version__)


class DeepSpeedMIIHandler(BaseHandler, ABC):
    """
    Diffusers handler class for text to image generation.
    """

    def __init__(self):
        self.initialized = False

    def initialize(self, ctx):
        """In this initialize function, the Stable Diffusion model is loaded and
        initialized here.
        Args:
            ctx (context): It is a JSON Object containing information
            pertaining to the model artefacts parameters.
        """
        self.manifest = ctx.manifest
        properties = ctx.system_properties
        model_dir = properties.get("model_dir")

        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id"))
            if torch.cuda.is_available() and properties.get("gpu_id") is not None
            else "cpu"
        )
        with zipfile.ZipFile(model_dir + "/model.zip", "r") as zip_ref:
            zip_ref.extractall(model_dir + "/model")

        # read configs for the mode, model_name, etc. from setup_config.json
        setup_config_path = os.path.join(model_dir, "setup_config.json")
        if os.path.isfile(setup_config_path):
            with open(setup_config_path) as setup_config_file:
                self.setup_config = json.load(setup_config_file)
        else:
            logger.warning("Missing the setup_config.json file.")

        provider = mii.constants.ModelProvider[self.setup_config["provider"].upper()]
        mii_configs = mii.MIIConfig(**self.setup_config["mii_configs"])
        self.pipe = mii.models.load_models(
            task_name=self.setup_config["task_name"],
            model_name=model_dir + "/model",
            model_path=model_dir + "/model",
            ds_optimize=self.setup_config["ds_optimize"],
            ds_zero=self.setup_config["ds_zero"],
            provider=provider,
            mii_config=mii_configs,
        )
        self.pipe.to(self.device)
        logger.info("Model from path %s loaded successfully", model_dir)

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
        # Handling inference for sequence_classification.
        generator = torch.Generator(self.device).manual_seed(1024)
        inferences = self.pipe(
            inputs,
            guidance_scale=self.setup_config["model_config"]["guidance_scale"],
            num_inference_steps=self.setup_config["model_config"][
                "num_inference_steps"
            ],
            generator=generator,
        ).images

        logger.info("Generated image: '%s'", inferences)
        return inferences

    def postprocess(self, inference_output):
        """Post Process Function converts the generated image into Torchserve readable format.
        Args:
            inference_output (list): It contains the generated image of the input text.
        Returns:
            (list): Returns a list of the images.
        """
        images = []
        for image in inference_output:
            images.append(np.array(image).tolist())
        return images
