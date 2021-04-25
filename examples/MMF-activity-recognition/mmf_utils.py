from abc import ABC
import json
import logging
import os
import pickle
import sys
import tempfile
import ast
import torch
import io
import PIL
from PIL import Image
import tempfile
import torchaudio
import torchvision
from omegaconf import OmegaConf

from torchvision import transforms
from torchvision.datasets.vision import VisionDataset

from torchvision.io import (
    read_video_timestamps,
    read_video
)
logger = logging.getLogger(__name__)

import csv
from transformers import BertTokenizer

sys.path.append("/home/ubuntu/mmf")

import transforms as T
PRE_TRAINED_MODEL_NAME = 'bert-base-uncased'

BERT_MAX_LEN = 112

bert_tokenizer_tfm = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME).encode_plus
class TruncateOrPad(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        if sample.shape[1]>= self.output_size:
            return sample[0,:self.output_size]
        else:
            return torch.cat(
                (
                    sample[0,:],
                    torch.zeros(1, output_size-sample.shape[1])
                ),
                axis=1
            )

def audio_proccessor(audio):

    audio_transform = transforms.Compose([
        TruncateOrPad(1000),
        torchaudio.transforms.MelSpectrogram(),
        transforms.ToPILImage(),
        transforms.Resize(224),
        #             transforms.CenterCrop(224),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return audio_transform(audio)

def video_processor(video):
    # video transforms
    normalize = T.Normalize(
        mean=[0.43216, 0.394666, 0.37645],
        std=[0.22803, 0.22145, 0.216989]
    )
    video_transform = torchvision.transforms.Compose([
        T.ToFloatTensorInZeroOne(),
        T.Resize((128, 171)),
        normalize,
        T.CenterCrop((112, 112))
    ])

    return video_transform(video)

def text_processor(script):
    max_seq_length = 128
    inputs = bert_tokenizer_tfm(script,max_length = max_seq_length ,pad_to_max_length = True, add_special_tokens = True, return_tensors = 'pt')
    return inputs

def video_audio_handler(video, script, device):
    import torchvision.io
    # self.init_processors()
    with tempfile.TemporaryDirectory() as dirname:
        extension = "mp4"
        fname = os.path.join(dirname,f"file.{extension}")
        with open(fname, "wb") as stream:
            stream.write(video)
        video_list =[fname]

        video_tensor, audio_tensor,info = torchvision.io.read_video(fname)
        video_tensor = video_tensor
        audio_tensor = audio_tensor
        video_transfomred = video_processor(video_tensor)
        audio_transfomred = audio_proccessor(audio_tensor)
        text_tensor = text_processor(script)
        for key in text_tensor.keys():
            text_tensor[key] = text_tensor[key].to(device)
        return video_transfomred.to(device),audio_transfomred.to(device),text_tensor

