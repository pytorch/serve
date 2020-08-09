from ts.torch_handler.base_handler import BaseHandler
from fairseq.models.transformer import TransformerModel
import torch
from pathlib import Path
import os
import logging

logger = logging.getLogger(__name__)


class LanguageTranslationHandler(BaseHandler):

    def __init__(self):
        self._context = None
        self.initialized = False
        self.model = None
        self.device = None

    def initialize(self, context):
        self._context = context
        self.initialized = True
        self.manifest = context.manifest

        properties = context.system_properties
        model_dir = properties.get("model_dir")

        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")

        #  load the model
        self.model = TransformerModel.from_pretrained(
            model_dir,
            checkpoint_file='model.pt',
            data_name_or_path=model_dir,
            tokenizer='moses',
            bpe='subword_nmt'
        )
        self.model.to(self.device)
        self.model.eval()

        self.initialized = True

    def preprocess(self, data):
        text = data[0].get("data")
        if text is None:
            text = data[0].get("body")
        input_text = text.decode('utf-8')
        return input_text

    def inference(self, data):
        translation = self.model.translate(data, beam=5)
        translation += '\n'
        logger.info("Model translated: '%s'", translation)
        return translation

    def postprocess(self, data):
        return [data]