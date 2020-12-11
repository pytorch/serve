from ts.torch_handler.base_handler import BaseHandler
from fairseq.models.transformer import TransformerModel
import torch
import json
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

        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")

        #read configs for the model_name, bpe etc. from setup_config.json
        setup_config_path = os.path.join(model_dir, "setup_config.json")
        if os.path.isfile(setup_config_path):
            with open(setup_config_path) as setup_config_file:
                self.setup_config = json.load(setup_config_file)
        else:
            logger.warning('Missing the setup_config.json file.')

        #  load the model
        self.model = TransformerModel.from_pretrained(
            model_dir,
            checkpoint_file='model.pt',
            data_name_or_path=model_dir,
            tokenizer='moses',
            bpe=self.setup_config["bpe"]
        )
        self.model.to(self.device)
        self.model.eval()
        self.initialized = True

    def preprocess(self, data):
        textInput = []
        for row in data:
            text = row.get("data") or row.get("body")
            decoded_text = text.decode('utf-8')
            textInput.append(decoded_text)
        return textInput

    def inference(self, data, *args, **kwargs):
        inference_output = []
        with torch.no_grad():
            translation = self.model.translate(data, beam=5)
        logger.info("Model translated: %s", translation)
        for i in range(0, len(data)):
            output = {
                "english_input": data[i],
                self.setup_config["translated_output"]: translation[i]
            }
            inference_output.append(json.dumps(output))
        return inference_output

    def postprocess(self, data):
        return data
