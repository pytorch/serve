import logging
import numpy as np
import os
import torch
import uuid
import zipfile
from waveglow_model import WaveGlow
from scipy.io.wavfile import write, read

logger = logging.getLogger(__name__)


class WaveGlowSpeechSynthesizer(object):

    def __init__(self):
        self.waveglow_model = None
        self.tacotron2_model = None
        self.mapping = None
        self.device = None
        self.initialized = False
        self.metrics = None

    # From https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/SpeechSynthesis/Tacotron2/inference.py
    def _unwrap_distributed(self, state_dict):
        """
        Unwraps model from DistributedDataParallel.
        DDP wraps model in additional "module.", it needs to be removed for single
        GPU inference.
        :param state_dict: model's state dict
        """
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace('module.', '')
            new_state_dict[new_key] = value
        return new_state_dict

    def _load_tacotron2_model(self, model_dir):
        from PyTorch.SpeechSynthesis.Tacotron2.tacotron2 import model as tacotron2
        from PyTorch.SpeechSynthesis.Tacotron2.tacotron2.text import text_to_sequence
        tacotron2_checkpoint = torch.load(os.path.join(model_dir, 'nvidia_tacotron2pyt_fp32_20190306.pth'))
        tacotron2_state_dict = self._unwrap_distributed(tacotron2_checkpoint['state_dict'])
        tacotron2_config = tacotron2_checkpoint['config']
        self.tacotron2_model = tacotron2.Tacotron2(**tacotron2_config)
        self.tacotron2_model.load_state_dict(tacotron2_state_dict)
        self.tacotron2_model.text_to_sequence = text_to_sequence
        self.tacotron2_model.to(self.device)

    def initialize(self, ctx):
        """First try to load torchscript else load eager mode state_dict based model"""

        properties = ctx.system_properties
        model_dir = properties.get("model_dir")
        if not torch.cuda.is_available():
            raise RuntimeError("This model is not supported on CPU machines.")
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")))

        with zipfile.ZipFile(model_dir + '/tacotron.zip', 'r') as zip_ref:
            zip_ref.extractall(model_dir)

        waveglow_checkpoint = torch.load(os.path.join(model_dir, "nvidia_waveglowpyt_fp32_20190306.pth"))
        waveglow_state_dict = self._unwrap_distributed(waveglow_checkpoint['state_dict'])
        waveglow_config = waveglow_checkpoint['config']
        self.waveglow_model = WaveGlow(**waveglow_config)
        self.waveglow_model.load_state_dict(waveglow_state_dict)
        self.waveglow_model = self.waveglow_model.remove_weightnorm(self.waveglow_model)
        self.waveglow_model.to(self.device)
        self.waveglow_model.eval()

        self._load_tacotron2_model(model_dir)

        logger.debug('WaveGlow model file loaded successfully')
        self.initialized = True

    def preprocess(self, data):
        """
         Scales, crops, and normalizes a PIL image for a MNIST model,
         returns an Numpy array
        """
        text = data[0].get("data")
        if text is None:
            text = data[0].get("body")
        text = text.decode('utf-8')

        sequence = np.array(self.tacotron2_model.text_to_sequence(text, ['english_cleaners']))[None, :]
        sequence = torch.from_numpy(sequence).to(device=self.device, dtype=torch.int64)

        return sequence

    def inference(self, data):
        with torch.no_grad():
            _, mel, _, _ = self.tacotron2_model.infer(data)
            audio = self.waveglow_model.infer(mel)

            return audio

    def postprocess(self, inference_output):
        audio_numpy = inference_output[0].data.cpu().numpy()
        path = "/tmp/{}.wav".format(uuid.uuid4().hex)
        write(path, 22050, audio_numpy)
        with open(path, 'rb') as output:
            data = output.read()
        os.remove(path)
        return [data]


_service = WaveGlowSpeechSynthesizer()


def handle(data, context):
    if not _service.initialized:
        _service.initialize(context)

    if data is None:
        return None

    data = _service.preprocess(data)
    data = _service.inference(data)
    data = _service.postprocess(data)

    return data
