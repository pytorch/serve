import logging
import os
from pathlib import Path

import numpy as np
import torch
from pipeline_utils import load_pipeline

from ts.handler_utils.timer import timed
from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)


class DiffusionFastHandler(BaseHandler):
    """
    Diffusion-Fast handler class for text to image generation.
    """

    def __init__(self):
        super().__init__()
        self.initialized = False

    def initialize(self, ctx):
        """In this initialize function, the Diffusion Fast model is loaded and
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

        # Parameters for the model
        compile_unet = ctx.model_yaml_config["handler"]["compile_unet"]
        compile_vae = ctx.model_yaml_config["handler"]["compile_vae"]
        compile_mode = ctx.model_yaml_config["handler"]["compile_mode"]
        enable_fused_projections = ctx.model_yaml_config["handler"][
            "enable_fused_projections"
        ]
        do_quant = ctx.model_yaml_config["handler"]["do_quant"]
        change_comp_config = ctx.model_yaml_config["handler"]["change_comp_config"]
        no_sdpa = ctx.model_yaml_config["handler"]["no_sdpa"]
        no_bf16 = ctx.model_yaml_config["handler"]["no_bf16"]
        upcast_vae = ctx.model_yaml_config["handler"]["upcast_vae"]

        # Load model weights
        model_path = Path(ctx.model_yaml_config["handler"]["model_path"])
        ckpt = os.path.join(model_dir, model_path)

        self.pipeline = load_pipeline(
            ckpt=ckpt,
            compile_unet=compile_unet,
            compile_vae=compile_vae,
            compile_mode=compile_mode,
            enable_fused_projections=enable_fused_projections,
            do_quant=do_quant,
            change_comp_config=change_comp_config,
            no_bf16=no_bf16,
            no_sdpa=no_sdpa,
            upcast_vae=upcast_vae,
        )

        logger.info("Diffusion Fast model loaded successfully")

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
        ), "Diffusion Fast is currently only supported with batch_size=1"

        inputs = []
        for _, data in enumerate(requests):
            input_text = data.get("data")
            if input_text is None:
                input_text = data.get("body")
            if isinstance(input_text, (bytes, bytearray)):
                input_text = input_text.decode("utf-8")
            inputs.append(input_text)
        return inputs

    @timed
    def inference(self, inputs):
        """Generates the image relevant to the received text.
        Args:
            input_batch (list): List of Text from the pre-process function is passed here
        Returns:
            list : It returns a list of the generate images for the input text
        """
        # Handling inference for sequence_classification.
        inferences = self.pipeline(
            inputs, num_inference_steps=self.num_inference_steps, height=768, width=768
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
