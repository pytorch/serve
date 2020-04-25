import io
import logging
import numpy as np
import os
import torch
from torch.autograd import Variable
from transformers import BertTokenizer, BertForSequenceClassification
import json




logger = logging.getLogger(__name__)


class Bertseqclassifier(object):
    """
    Bertseqclassifier handler class. This handler takes a piece of text
    and returns the classification prediction which here is 4 classes in ag_news as presented in
    index_to_name.json.
    """

    def __init__(self):
        self.model = None
        self.mapping = None
        self.device = None
        self.initialized = False



    def initialize(self, ctx):
        """Loading the serialized file/ pre-trained model from transfromers here is bert-base model"""

        properties = ctx.system_properties
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")
        model_dir = properties.get("model_dir")

        # Read model serialize/pt file and if not found load it from pe-trained model
        model_pt_path = os.path.join(model_dir, "traced_bert.pt")
        if not os.path.isfile(model_pt_path):
            self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels = 4)
        self.model = torch.jit.load(model_pt_path)

        self.model.to(self.device)
        self.model.eval()
        # Read the mapping file, index to object name
        mapping_file_path = os.path.join(model_dir, "index_to_name.json")

        if os.path.isfile(mapping_file_path):
            with open(mapping_file_path) as f:
                self.mapping = json.load(f)
        else:
            logger.warning('Missing the index_to_name.json file. Inference output will not include class name.')
        # logger.debug('Model file {0} loaded successfully'.format(model_pt_path))
        self.initialized = True

    def preprocess(self, data):
        """
         Process the text to the input format for Bert model.
        """
        text = data[0].get("data")
        if text is None:
            text = data[0].get("body")

        # Convert to a "unicode" object
        text_obj = text.decode('UTF-8')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        input_ids = torch.tensor(self.tokenizer.encode(text_obj, add_special_tokens=True)).unsqueeze(0)  # Batch size 1


        return input_ids

    def inference(self, input_ids, topk=5):
        ''' Predict the class (or classes) of the received text using the BertForSequenceClassification model.
        '''


        self.model.eval()
        outputs = self.model(input_ids)
        output = outputs[0].argmax(1).item() + 1
        if self.mapping:
            output = self.mapping[str(output)]

        return [output]

    def postprocess(self, inference_output):

        return inference_output


_service = Bertseqclassifier()


def handle(data, context):
    if not _service.initialized:
        _service.initialize(context)

    if data is None:
        return None

    data = _service.preprocess(data)
    data = _service.inference(data)
    data = _service.postprocess(data)

    return data
