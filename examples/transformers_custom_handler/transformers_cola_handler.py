from abc import ABC
import json
import logging
import os
from uuid import uuid4

import numpy as np
from sklearn.utils.extmath import softmax
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.data.processors.glue import glue_convert_examples_to_features
from transformers.data.processors.utils import InputExample
from transformers import glue_processors as processors

from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)



class TransformersClassifierHandler(BaseHandler, ABC):
    """
    Transformers text classifier handler class. This handler takes a text (string) and
    as input and returns the classification text based on the serialized transformers checkpoint.
    """
    def __init__(self):
        super(TransformersClassifierHandler, self).__init__()
        self.initialized = False

    def initialize(self, ctx):
        self.manifest = ctx.manifest

        properties = ctx.system_properties
        model_dir = properties.get("model_dir")
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")

        # Read model serialize/pt file
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

        self.model.to(self.device)
        self.model.eval()

        logger.debug('_Transformer model from path {0} loaded successfully'.format(model_dir))

        # Read the mapping file, index to object name
        mapping_file_path = os.path.join(model_dir, "index_to_name.json")

        if os.path.isfile(mapping_file_path):
            with open(mapping_file_path) as f:
                self.mapping = json.load(f)
        else:
            logger.warning('Missing the index_to_name.json file. Inference output will not include class name.')

        self.initialized = True

    def preprocess(self, data):
        """ Very basic preprocessing code - only tokenizes.
            Extend with your own preprocessing steps as needed.
        """
        text = data[0].get("data")
        if text is None:
            text = data[0].get("body")
        text = text.decode('utf-8')
        logger.info("Received text: '%s'", text)

        inputs = self.tokenizer.encode_plus(text, max_length=128, pad_to_max_length=True, return_tensors="pt")
        return inputs

    def inference(self, inputs):
        """
        Predict the class of a text using a trained transformer model.
        """
        with torch.no_grad():
            inputs = {input_name: input_values.to(self.device) for input_name, input_values in inputs.items()}
            outputs, *_ = self.model(**inputs)

        outputs = softmax(outputs.numpy())
        prediction, = int(np.argmax(outputs, axis=1))

        if self.mapping:
            prediction = self.mapping[str(prediction)]

        return [prediction]

    def postprocess(self, inference_output):
        # TODO: Add any needed post-processing of the model predictions here
        return inference_output


_service = TransformersClassifierHandler()


def handle(data, context):
    try:
        if not _service.initialized:
            _service.initialize(context)

        if data is None:
            return None

        data = _service.preprocess(data)
        data = _service.inference(data)
        data = _service.postprocess(data)

        return data
    except Exception as e:
        raise e
