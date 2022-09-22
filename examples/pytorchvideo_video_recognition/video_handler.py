"""
Handler for PyTorchVideo Video Recognition Example
"""
import io
import json
import logging
import os
from abc import ABC

import torch
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
)
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import CenterCropVideo, NormalizeVideo
from transform_config import PackPathway

from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)


class PyTorchVideoHandler(BaseHandler, ABC):
    """
    Handler for TorchRec DLRM example
    """

    def initialize(self, context):
        """Initialize function loads the model.pt file and initialized the model object.
           This version creates and initialized the model on cpu fist and transfers to gpu in a second step to prevent GPU OOM.
        Args:
            context (context): It is a JSON Object containing information
            pertaining to the model artifacts parameters.
        Raises:
            RuntimeError: Raises the Runtime error when the model.py is missing
        """
        properties = context.system_properties

        # Set device to cpu to prevent GPU OOM errors
        self.device = "cpu"
        self.manifest = context.manifest

        model_dir = properties.get("model_dir")
        model_pt_path = None
        if "serializedFile" in self.manifest["model"]:
            serialized_file = self.manifest["model"]["serializedFile"]
            model_pt_path = os.path.join(model_dir, serialized_file)

        # model def file
        model_file = self.manifest["model"].get("modelFile", "")

        if not model_file:
            raise RuntimeError("model.py not specified")

        logger.debug("Loading eager model")
        self.model = self._load_pickled_model(model_dir, model_file, model_pt_path)

        self.map_location = (
            "cuda"
            if torch.cuda.is_available() and properties.get("gpu_id") is not None
            else "cpu"
        )

        self.device = torch.device(
            self.map_location + ":" + str(properties.get("gpu_id"))
            if torch.cuda.is_available() and properties.get("gpu_id") is not None
            else self.map_location
        )

        self.model.to(self.device)

        self.model.eval()

        logger.debug("Model file %s loaded successfully", model_pt_path)

        with open("kinetics_classnames.json", "r") as f:
            kinetics_classnames = json.load(f)

        # Create an id to label name mapping
        self.kinetics_id_to_classname = {}
        for k, v in kinetics_classnames.items():
            self.kinetics_id_to_classname[v] = str(k).replace('"', "")

        self.initialized = True

    def preprocess(self, data):
        """
        The input values for the DLRM model are twofold. There is a dense part and a sparse part.
        The sparse part consists of a list of ids where each entry can consist of zero, one or multiple ids.
        Due to the inconsistency in elements, the sparse part is represented by the KeyJaggedTensor class provided by TorchRec.

        Args:
            data (str): The input data is in the form of a string

        Returns:
            Tuple of:
                (Tensor): Dense features
                (KeyJaggedTensor): Sparse features
        """

        float_features, id_list_features_lengths, id_list_features_values = [], [], []

        ####################
        # SlowFast transform
        ####################

        side_size = 256
        mean = [0.45, 0.45, 0.45]
        std = [0.225, 0.225, 0.225]
        crop_size = 256
        num_frames = 32
        sampling_rate = 2
        frames_per_second = 30
        alpha = 4

        transform = ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(num_frames),
                    Lambda(lambda x: x / 255.0),
                    NormalizeVideo(mean, std),
                    ShortSideScale(size=side_size),
                    CenterCropVideo(crop_size),
                    PackPathway(),
                ]
            ),
        )

        for row in data:

            input = row.get("data") or row.get("body")

            # if not isinstance(input, dict):
            #    input = json.loads(input)

            video = io.BytesIO(input)

            # The duration of the input clip is also specific to the model.
            clip_duration = (num_frames * sampling_rate) / frames_per_second

            # Select the duration of the clip to load by specifying the start and end duration
            # The start_sec should correspond to where the action occurs in the video
            start_sec = 0
            end_sec = start_sec + clip_duration

            # Initialize an EncodedVideo helper class
            video = EncodedVideo.from_binary(video, "inp_video")

            # Load the desired clip
            video_data = video.get_clip(start_sec=start_sec, end_sec=end_sec)

            # Apply a transform to normalize the video input
            video_data = transform(video_data)

            # Move the inputs to the desired device
            inputs = video_data["video"]
            inputs = [i.to(self.device)[None, ...] for i in inputs]
            print("Done preprocess")

        return inputs

    def inference(self, data):
        """
        The inference call moves the elements of the tuple onto the device and calls the model

        Args:
            data (torch tensor): The data is in the form of Torch Tensor
                                 whose shape should match that of the
                                  Model Input shape.

        Returns:
            (Torch Tensor): The predicted response from the model is returned
                            in this function.
        """
        with torch.no_grad():
            # data = map(lambda x: x.to(self.device), data)
            results = self.model(data)

        return results

    def postprocess(self, data):
        """
        The post process function converts the prediction response into a
           Torchserve compatible format

        Args:
            data (Torch Tensor): The data parameter comes from the prediction output
            output_explain (None): Defaults to None.

        Returns:
            (list): Returns the response containing the predictions which consist of a single score per input entry.
        """

        # Get the predicted classes
        post_act = torch.nn.Softmax(dim=1)
        preds = post_act(data)
        pred_classes = preds.topk(k=5).indices

        # Map the predicted classes to the label names
        pred_class_names = [
            self.kinetics_id_to_classname[int(i)] for i in pred_classes[0]
        ]

        return [pred_class_names]
