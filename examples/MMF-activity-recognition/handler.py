import json
import logging
import os
import pickle
import sys
import ast
import torch
from ts.torch_handler.base_handler import BaseHandler
import io
import torchaudio
import torchvision
from omegaconf import OmegaConf
import pandas as pd
import csv

from torchvision import transforms
from torchvision.datasets.vision import VisionDataset
from torchvision.io import (
    read_video_timestamps,
    read_video
)
logger = logging.getLogger(__name__)

from mmf.common.sample import Sample, SampleList
from mmf.utils.env import set_seed, setup_imports
from mmf.utils.logger import setup_logger, setup_very_basic_config
from mmf.datasets.base_dataset import BaseDataset
from mmf.utils.build import build_encoder, build_model, build_processors
from mmf.datasets.mmf_dataset_builder import MMFDatasetBuilder
from torch.utils.data import IterableDataset
from mmf.utils.configuration import load_yaml
from mmf.models.mmf_transformer import MMFTransformer

class MMFHandler(BaseHandler):
    """
    Transformers handler class for  MMFTransformerWithVideoAudio model.
    """
    def __init__(self):
        super(MMFHandler, self).__init__()
        self.initialized = False

    def initialize(self, ctx):
        self.manifest = ctx.manifest
        properties = ctx.system_properties
        model_dir = properties.get("model_dir")
        serialized_file = self.manifest['model']['serializedFile']
        model_pt_path = os.path.join(model_dir, serialized_file)
        self.map_location = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(
            self.map_location + ":" + str(properties.get("gpu_id"))
            if torch.cuda.is_available()
            else self.map_location
        )

        # reading the csv file which include all the labels in the dataset to make the class/index mapping
        # and matching the output of the model with num labels from dataset
        df = pd.read_csv('./charades_action_lables.csv')
        label_set = set()
        df['action_labels'] = df['action_labels'].str.replace('"','')
        labels_initial = df['action_labels'].tolist()
        labels = []
        for sublist in labels_initial:
            new_sublist = ast.literal_eval(sublist)
            labels.append(new_sublist)
            for item in new_sublist:
                label_set.add(item)
        classes = sorted(list(label_set))
        self.class_to_idx = {classes[i]: i for i in range(len(classes))}
        self.classes = classes
        self.labels = labels
        self.idx_to_class = classes
        config = OmegaConf.load('config.yaml')
        print("*********** config keyssss **********", config.keys())
        setup_very_basic_config()
        setup_imports()
        self.model = MMFTransformer(config.model_config.mmf_transformer)
        self.model.build()
        self.model.init_losses()
        self.processor = build_processors(
            config.dataset_config["charades"].processors
        )
        state_dict = torch.load(serialized_file, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        self.initialized = True
        print("********* files in temp direcotry that .mar file got extracted *********", os.listdir(model_dir))

    def preprocess(self, requests):
        """ Preprocessing, based on processor defined for MMF model.
            """

        def create_sample(video_transfomred,audio_transfomred,text_tensor, video_label):

            label = [self.class_to_idx[l] for l in video_label]

            one_hot_label = torch.zeros(len(self.class_to_idx))
            one_hot_label[label] = 1

            current_sample= Sample()
            current_sample.video = video_transfomred
            current_sample.audio = audio_transfomred
            current_sample.update(text_tensor)
            current_sample.targets = one_hot_label
            current_sample.dataset_type = 'test'
            current_sample.dataset_name = 'charades'
            return SampleList([current_sample]).to(self.device)

        for idx, data in enumerate(requests):
            raw_script = data.get('script')
            script = raw_script.decode('utf-8')
            raw_label = data.get('labels')
            video_label = raw_label.decode('utf-8')
            video_label = [video_label]
            
            video = io.BytesIO(data['data'])
            video_tensor, audio_tensor,info = torchvision.io.read_video(video)
            text_tensor = self.processor["text_processor"]({"text": script})
            video_transformed = self.processor["video_test_processor"](video_tensor)
            audio_transformed = self.processor["audio_processor"](audio_tensor)
            samples = create_sample(video_transformed,audio_transformed,text_tensor,video_label)

        return samples

    def inference(self, samples):
        """ Predict the class (or classes) of the received text using the serialized transformers checkpoint.
        """
        if torch.cuda.is_available():
            with torch.cuda.device(samples.get_device()):
                output = self.model(samples)
        else:
            output = self.model(samples)
            
        sigmoid_scores = torch.sigmoid(output["scores"])
        binary_scores = torch.round(sigmoid_scores)
        score = binary_scores[0]
        score = score.nonzero()

        predictions = []
        for item in score:
            predictions.append(self.idx_to_class[item.item()])
        print("************** predictions *********", predictions)
        return predictions

    def postprocess(self, inference_output):
        # TODO: Add any needed post-processing of the model predictions here
        return [inference_output]
