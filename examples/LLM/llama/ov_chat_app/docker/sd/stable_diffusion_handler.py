import logging
import os
from pathlib import Path

import numpy as np
import torch
from pipeline_utils import load_pipeline
import json

from ts.handler_utils.timer import timed
from ts.torch_handler.base_handler import BaseHandler
from ts.utils.util import check_valid_pt2_backend


logger = logging.getLogger(__name__)


class StableDiffusionHandler(BaseHandler):
    """
    StableDiffusion handler class for text to image generation.
    """

    def __init__(self):
        super().__init__()
        self.initialized = False

    def initialize(self, ctx):
        """In this initialize function, the Stable Diffusion model is loaded and
        initialized here.
        Args:
            ctx (context): It is a JSON Object containing information
            pertaining to the model artifacts parameters.
        """
        self.context = ctx
        self.manifest = ctx.manifest
        properties = ctx.system_properties
        model_dir = properties.get("model_dir")

        if torch.cuda.is_available() and properties.get("gpu_id") is not None:
            self.map_location = "cuda"
            self.device = torch.device(
                self.map_location + ":" + str(properties.get("gpu_id"))
            )
        else:
            self.map_location = "cpu"
            self.device = torch.device(self.map_location)

        self.num_inference_steps = ctx.model_yaml_config["handler"][
            "num_inference_steps"
        ]

        print(f"Config is {ctx.model_yaml_config}")

        # Parameters for the model
        compile_unet = ctx.model_yaml_config["handler"]["compile_unet"]
        compile_vae = ctx.model_yaml_config["handler"]["compile_vae"]
        compile_mode = ctx.model_yaml_config["handler"]["compile_mode"]
        change_comp_config = ctx.model_yaml_config["handler"]["change_comp_config"]
        is_xl = ctx.model_yaml_config["handler"]["is_xl"]

        compile_options = {}
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
        model_path = Path(ctx.model_yaml_config["handler"]["model_path"])
        ckpt = os.path.join(model_dir, model_path)

        self.pipeline = load_pipeline(
            ckpt=ckpt,
            compile_unet=compile_unet,
            compile_vae=compile_vae,
            compile_mode=compile_mode,
            change_comp_config=change_comp_config,
            compile_options=compile_options,
            is_xl=is_xl,
        )

        logger.info("Stable Diffusion model loaded successfully")

        self.initialized = True

    @timed
    def preprocess(self, requests):
        """Basic text preprocessing, of the user's prompt.
        Args:
            requests (str): The Input data in the form of text is passed on to the preprocess
            function.
        Returns:
            list : The preprocess function returns a list of prompts.
        """

        assert (
            len(requests) == 1
        ), "Stable Diffusion is currently only supported with batch_size=1"

        data = requests[0]
        input_text = data.get("data") or data.get("body")
        return input_text

    @timed
    def inference(self, model_inputs):
        """Generates the image relevant to the received text.
        Args:
            input_batch (list): List of Text from the pre-process function is passed here
        Returns:
            list : It returns a list of the generate images for the input text
        """
        # Handling inference for sequence_classification.
        model_inputs = json.loads(model_inputs)
        guidance_scale = model_inputs.get("guidance_scale") or 5.0
        num_inference_steps = model_inputs.get("num_inference_steps") or 30
        height = model_inputs.get("height") or 512
        width = model_inputs.get("width") or 512
        inferences = self.pipeline(
            model_inputs["prompt"], num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, height=height, width=width
        ).images

        return inferences

    @timed
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
