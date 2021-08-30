import os
import json
import sys
import logging

import torch, torch_neuron
from transformers import AutoTokenizer
from abc import ABC
from ts.torch_handler.base_handler import BaseHandler

# one core per worker
os.environ['NEURONCORE_GROUP_SIZES'] = '1'

logger = logging.getLogger(__name__)

class BertEmbeddingHandler(BaseHandler, ABC):
    """
    Handler class for Bert Embedding computations.
    """
    def __init__(self):
        super(BertEmbeddingHandler, self).__init__()
        self.initialized = False

    def initialize(self, ctx):
        self.manifest = ctx.manifest
        properties = ctx.system_properties
        self.device = 'cpu'
        model_dir = properties.get('model_dir')
        serialized_file = self.manifest['model']['serializedFile']
        model_pt_path = os.path.join(model_dir, serialized_file)

        # point sys.path to our config file
        sys.path.append(model_dir)
        import config
        self.max_length = config.max_length
        self.batch_size = config.batch_size
        self.classes = ['not paraphrase', 'paraphrase']

        self.model = torch.jit.load(model_pt_path)
        logger.debug(f'Model loaded from {model_dir}')
        self.model.to(self.device)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.initialized = True

    def preprocess(self, input_data):
        """
        Tokenization pre-processing
        """

        input_ids = []
        attention_masks = []
        token_type_ids = []

        for row in input_data:
            #seq_0 = row['body']['seq_0'].decode('utf-8')
            #seq_1 = row['body']['seq_1'].decode('utf-8')

            json_data = json.loads(row['body'].decode('utf-8'))

            seq_0 = json_data['seq_0']
            seq_1 = json_data['seq_1']
            logger.debug(f'Received text: "{seq_0}", "{seq_1}"')

            inputs = self.tokenizer.encode_plus(
                    seq_0,
                    seq_1,
                    max_length=self.max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                    )

            input_ids.append(inputs['input_ids'])
            attention_masks.append(inputs['attention_mask'])
            token_type_ids.append(inputs['token_type_ids'])

        batch = (torch.cat(input_ids, 0),
                torch.cat(attention_masks, 0),
                torch.cat(token_type_ids, 0))

        return batch

    def inference(self, inputs):
        """
        Predict the class of a text using a trained transformer model.
        """

        # sanity check dimensions
        assert(len(inputs) == 3)
        num_inferences = len(inputs[0])
        assert(num_inferences <= self.batch_size)

        # insert padding if we received a partial batch
        padding = self.batch_size - num_inferences
        if padding > 0:
            pad = torch.nn.ConstantPad1d((0, 0, 0, padding), value=0)
            inputs = [pad(x) for x in inputs]

        outputs = self.model(*inputs)[0]
        predictions = []
        for i in range(num_inferences):
            prediction = self.classes[outputs[i].argmax().item()]
            predictions.append([prediction])
            logger.debug("Model predicted: '%s'", prediction)
        return predictions

    def postprocess(self, inference_output):
        return inference_output