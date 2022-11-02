import torch
import torchaudio
from transformers import AutoProcessor, AutoModelForCTC
import io


class Wav2VecHandler(object):
    def __init__(self):
        self._context = None
        self.initialized = False
        self.model = None
        self.processor = None
        self.device = None
        # Sampling rate for Wav2Vec model must be 16k
        self.expected_sampling_rate = 16_000

    def initialize(self, context):
        """Initialize properties and load model"""
        self._context = context
        self.initialized = True
        properties = context.system_properties

        # See https://pytorch.org/serve/custom_service.html#handling-model-execution-on-multiple-gpus
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")

        model_dir = properties.get("model_dir")
        self.processor = AutoProcessor.from_pretrained(model_dir)
        self.model = AutoModelForCTC.from_pretrained(model_dir)

    def handle(self, data, context):
        """Transform input to tensor, resample, run model and return transcribed text."""
        input = data[0].get("data")
        if input is None:
            input = data[0].get("body")
        
        # torchaudio.load accepts file like object, here `input` is bytes
        model_input, sample_rate = torchaudio.load(io.BytesIO(input), format="WAV")
        
        # Ensure sampling rate is the same as the trained model
        if sample_rate != self.expected_sampling_rate:
            model_input = torchaudio.functional.resample(model_input, sample_rate, self.expected_sampling_rate)
        
        model_input = self.processor(model_input, sampling_rate = self.expected_sampling_rate, return_tensors="pt").input_values[0]
        logits = self.model(model_input)[0]
        pred_ids = torch.argmax(logits, axis=-1)[0]
        output = self.processor.decode(pred_ids)

        return [output]
