from abc import ABC
import json
import logging
import os
import pickle
import sys
import tempfile
import ast
import torch
from ts.torch_handler.base_handler import BaseHandler
import io
import PIL
from PIL import Image
import zipfile
import tempfile
import torchaudio
import torchvision
from omegaconf import OmegaConf
import pandas as pd
from pathlib import Path
from functools import partial
from tqdm import tqdm
import csv

from torchvision import transforms
from torchvision.datasets.vision import VisionDataset

from torchvision.io import (
    read_video_timestamps,
    read_video
)
logger = logging.getLogger(__name__)

from transformers.modeling_bert import BertPooler, BertPredictionHeadTransform

from mmf.common.registry import registry
import torchvision.datasets.folder as tv_helpers
from mmf.common.sample import Sample, SampleList
from mmf.utils.env import set_seed, setup_imports
from mmf.utils.logger import setup_logger, setup_very_basic_config

from mmf.datasets.base_dataset import BaseDataset
from mmf.datasets.mmf_dataset_builder import MMFDatasetBuilder
from transformers import BertTokenizer
from torch.utils.data import IterableDataset
from mmf.utils.configuration import load_yaml
from mmf.models.mmf_transformer import MMFTransformer
import transforms as T
import mmf_utils
from mmf_utils import audio_proccessor, video_processor, text_processor, video_audio_handler

class MMFHandler(BaseHandler, ABC):
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
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")

        # reading the csv file which include all the labels in the dataset to make the class/index mapping
        # and matching the output of the model with num labels from dataset
        df = pd.read_csv('./charades.csv')
        # print(df.head(2))
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
        state_dict = torch.load(serialized_file)
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
            text_tensor['input_mask'] = text_tensor['attention_mask']
            text_tensor['segment_ids'] = text_tensor['token_type_ids']

            del text_tensor['attention_mask']
            for key in text_tensor.keys():
                text_tensor[key]= text_tensor[key].squeeze(0)
            current_sample.update(text_tensor)
            current_sample.targets = one_hot_label.to(self.device)
            current_sample.dataset_type = 'test'
            current_sample.dataset_name = 'charades'
            return SampleList([current_sample])

        for idx, data in enumerate(requests):
            raw_script = data.get('script')
            script = raw_script.decode('utf-8')
            raw_label = data.get('lables')
            video_label = raw_label.decode('utf-8')
            video_label = [video_label]

            video_transfomred,audio_transfomred,text_tensor = video_audio_handler(data['data'], script,self.device)
            samples = create_sample(video_transfomred,audio_transfomred,text_tensor,video_label)

        return samples

    def inference(self, samples):
        """ Predict the class (or classes) of the received text using the serialized transformers checkpoint.
        """
        output = self.model(samples)

        sigmoind_scores = torch.sigmoid(output["scores"])
        binary_scores = torch.round(sigmoind_scores)
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



# _service = MMFHandler()


# def handle(data, context):
#     try:
#         if not _service.initialized:
#             _service.initialize(context)

#         if data is None:
#             return None

#         data = _service.preprocess(data)
#         data = _service.inference(data)
#         data = _service.postprocess(data)

#         return data
#     except Exception as e:
#         raise e
