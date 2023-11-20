import base64
import io
import json
import logging
import pickle
import zlib

import cv2
import numpy as np
import torch
from PIL import Image
from segment_anything_fast import SamAutomaticMaskGenerator, sam_model_registry
from segment_anything_fast.tools import apply_eval_dtype_predictor

from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)


class SegmentAnythingFastHandler(BaseHandler):
    def __init__(self):
        super(SegmentAnythingFastHandler, self).__init__()
        self.mask_generator = None
        self.initialized = False

    def initialize(self, ctx):
        properties = ctx.system_properties
        if torch.cuda.is_available() and properties.get("gpu_id") is not None:
            self.map_location = "cuda"
            self.device = torch.device(
                self.map_location + ":" + str(properties.get("gpu_id"))
            )

        model_type = ctx.model_yaml_config["handler"]["model_type"]
        sam_checkpoint = ctx.model_yaml_config["handler"]["sam_checkpoint"]

        self.model = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.model.to(self.device)

        self.mask_generator = SamAutomaticMaskGenerator(self.model)
        self.mask_generator.predictor = apply_eval_dtype_predictor(
            self.mask_generator.predictor, torch.bfloat16
        )

        logger.info(f"Model weights {sam_checkpoint} loaded successfully")
        self.initialized = True

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

    def inference(self, data):
        return self.mask_generator.generate(data[0])

    def postprocess(self, data):
        # Convert the mask array to a string object
        class NumpyArrayEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray):
                    # return obj.tolist()
                    return "__nparray__" + json.dumps(obj.tolist())
                return json.JSONEncoder.default(self, obj)

        json_data = json.dumps({"data": data}, cls=NumpyArrayEncoder)

        # Compress the string
        compressed_data = zlib.compress(json_data.encode("utf-8"))

        # Serialize the compressed data using Pickle
        serialized_data = pickle.dumps(compressed_data)

        # Encode the serialized data as Base64
        base64_encoded_data = base64.b64encode(serialized_data).decode("utf-8")

        return [base64_encoded_data]
