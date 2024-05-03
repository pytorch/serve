import logging
import os
import uuid

import soundfile as sf
import torch
from datasets import load_from_disk
from transformers import SpeechT5ForTextToSpeech, SpeechT5HifiGan, SpeechT5Processor

from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)


class SpeechT5_TTS(BaseHandler):
    def __init__(self):
        self.model = None
        self.processor = None
        self.vocoder = None
        self.speaker_embeddings = None
        self.output_dir = "/tmp"

    def initialize(self, ctx):
        properties = ctx.system_properties
        model_dir = properties.get("model_dir")

        processor = ctx.model_yaml_config["handler"]["processor"]
        model = ctx.model_yaml_config["handler"]["model"]
        vocoder = ctx.model_yaml_config["handler"]["vocoder"]
        embeddings_dataset = ctx.model_yaml_config["handler"]["speaker_embeddings"]
        self.output_dir = ctx.model_yaml_config["handler"]["output_dir"]

        self.processor = SpeechT5Processor.from_pretrained(processor)
        self.model = SpeechT5ForTextToSpeech.from_pretrained(model)
        self.vocoder = SpeechT5HifiGan.from_pretrained(vocoder)

        # load xvector containing speaker's voice characteristics from a dataset
        embeddings_dataset = load_from_disk(embeddings_dataset)
        self.speaker_embeddings = torch.tensor(
            embeddings_dataset[7306]["xvector"]
        ).unsqueeze(0)

    def preprocess(self, requests):
        assert len(requests) == 1, "This is currently supported with batch_size=1"
        req_data = requests[0]

        input_data = req_data.get("data") or req_data.get("body")

        if isinstance(input_data, (bytes, bytearray)):
            input_data = input_data.decode("utf-8")

        inputs = self.processor(text=input_data, return_tensors="pt")

        return inputs

    def inference(self, inputs):
        output = self.model.generate_speech(
            inputs["input_ids"], self.speaker_embeddings, vocoder=self.vocoder
        )
        return output

    def postprocess(self, inference_output):
        path = self.output_dir + "/{}.wav".format(uuid.uuid4().hex)
        sf.write(path, inference_output.numpy(), samplerate=16000)
        with open(path, "rb") as output:
            data = output.read()
        os.remove(path)
        return [data]
