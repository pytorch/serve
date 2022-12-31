import json
import os
import tempfile

import torch
import torchaudio
from torchaudio.pipelines import EMFORMER_RNNT_BASE_LIBRISPEECH


class ModelHandler(object):
    """
    A custom model handler implementation.
    """
    def __init__(self):
        self._context = None
        self.initialized = False
        self.model = None
        self.device = None

    def initialize(self, context):
        """
        Invoke by torchserve for loading a model
        :param context: context contains model server system properties
        :return:
        """

        #  load the model
        self.manifest = context.manifest

        properties = context.system_properties
        model_dir = properties.get("model_dir")
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")

        # Read model serialize/pt file
        serialized_file = self.manifest['model']['serializedFile']
        model_pt_path = os.path.join(model_dir, serialized_file)
        if not os.path.isfile(model_pt_path):
            raise RuntimeError("Missing the model.pt file")

        self.model = torch.jit.load(model_pt_path)

        self.initialized = True


    def handle(self, data, context):
        """
        Invoke by TorchServe for prediction request.
        Do pre-processing of data, prediction using model and postprocessing of prediciton output
        :param data: Input data for prediction
        :param context: Initial context contains model server system properties.
        :return: prediction output
        """
        if isinstance(data, list):
            data = data[0]
        data = data.get("data") or data.get("body")

        # print('55-------', type(data)) # should be bytearray

        fp = tempfile.TemporaryFile()
        fp.write(data)
        fp.seek(0)

        waveform, sample_rate = torchaudio.load(fp)
        waveform = waveform.squeeze()

        feature_extractor = EMFORMER_RNNT_BASE_LIBRISPEECH.get_feature_extractor()
        
        decoder = self.model
        
        token_processor = EMFORMER_RNNT_BASE_LIBRISPEECH.get_token_processor()

        with torch.no_grad():
            features, length = feature_extractor(waveform)
            hypos = decoder(features, length, 1)

        # return batch_size = 1
        return [token_processor(hypos[0].tokens)]