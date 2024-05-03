import base64
import io
import logging
import os
import pickle

import cv2
import numpy as np
import torch
from PIL import Image
from segment_anything_fast import SamAutomaticMaskGenerator, sam_model_fast_registry

from ts.handler_utils.timer import timed
from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)


class SegmentAnythingFastHandler(BaseHandler):
    def __init__(self):
        super().__init__()
        self.mask_generator = None
        self.initialized = False

    def initialize(self, ctx):
        properties = ctx.system_properties
        model_dir = properties.get("model_dir")
        self.device = "cpu"
        if torch.cuda.is_available() and properties.get("gpu_id") is not None:
            self.map_location = "cuda"
            self.device = torch.device(
                self.map_location + ":" + str(properties.get("gpu_id"))
            )
            torch.cuda.set_device(self.device)

        model_type = ctx.model_yaml_config["handler"]["model_type"]
        sam_checkpoint = os.path.join(
            model_dir, ctx.model_yaml_config["handler"]["sam_checkpoint"]
        )
        process_batch_size = ctx.model_yaml_config["handler"]["process_batch_size"]

        self.model = sam_model_fast_registry[model_type](checkpoint=sam_checkpoint)
        self.model.to(self.device)

        self.mask_generator = SamAutomaticMaskGenerator(
            self.model, process_batch_size=process_batch_size, output_mode="coco_rle"
        )

        logger.info(
            f"Model weights {sam_checkpoint} for {model_type} loaded successfully with process batch size {process_batch_size}"
        )
        self.initialized = True

    @timed
    def preprocess(self, data):
        images = []
        for row in data:
            image = row.get("data") or row.get("body")
            if isinstance(image, str):
                # if the image is a string of bytesarray.
                image = base64.b64decode(image)

            # If the image is sent as bytesarray
            if isinstance(image, (bytearray, bytes)):
                image = Image.open(io.BytesIO(image))
            else:
                # if the image is a list
                image = torch.FloatTensor(image)

            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            images.append(image)

        return images

    @timed
    def inference(self, data):
        assert (
            len(data) == 1
        ), "SAM AutoMaticMaskGenerator currently supports batch size of 1"
        return self.mask_generator.generate(data[0])

    @timed
    def postprocess(self, data):
        # Serialize the output using Pickle
        serialized_data = pickle.dumps(data)

        # Encode the serialized data as Base64
        base64_encoded_data = base64.b64encode(serialized_data).decode("utf-8")

        return [base64_encoded_data]
