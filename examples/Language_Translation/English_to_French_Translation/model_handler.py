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
        """
        Invoke by torchserve for loading a model
        :param context: context contains model server system properties
        :return:
        """
        self._context = context
        self.initialized = True
        self.manifest = context.manifest

        #  load the model
        self.model = TransformerModel.from_pretrained(
            '/home/ashwin/Eng2Fr_Translation/wmt14.en-fr.joined-dict.transformer/',
            checkpoint_file='model.pt',
            data_name_or_path='/home/ashwin/Eng2Fr_Translation/wmt14.en-fr.joined-dict.transformer/',
            tokenizer='moses',
            bpe='subword_nmt'
        )

        self.initialized = True

    def preprocess(self, data):
        """ Basic text preprocessing, based on the user's chocie of application mode.
        """
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