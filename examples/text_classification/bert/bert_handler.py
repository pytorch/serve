import io
import logging
import numpy as np
import os
import torch
from torch.autograd import Variable
from transformers import BertTokenizer, BertForSequenceClassification


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
        """We simply load the pre-trained model from transfromers here is bert-base model"""

        properties = ctx.system_properties
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")
        model_dir = properties.get("model_dir")

        # Read model serialize/pt file and if not found load it from pe-trained model 
        model_pt_path = os.path.join(model_dir, "traced_bert.pt")
        if not os.path.isfile(model_pt_path):
            self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels = 4)
        self.model = torch.jit.load(model_pt_path)

        # Read model definition file
        # model_def_path = os.path.join(model_dir, "modelbert.py")
        # if not os.path.isfile(model_def_path):
        #     raise RuntimeError("Missing the model definition file")

        # self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels = 4)
        self.model.to(self.device)
        self.model.eval()

        # logger.debug('Model file {0} loaded successfully'.format(model_pt_path))
        self.initialized = True

    def preprocess(self, data):
        """
         Process the text to the input format for Bert model.
        """
        text = data[0].get("data")
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        input_ids = torch.tensor(self.tokenizer.encode(text, add_special_tokens=True)).unsqueeze(0)  # Batch size 1


        return input_ids

    def inference(self, input_ids, topk=5):
        ''' Predict the class (or classes) of the received text using the BertForSequenceClassification model.
        '''


        self.model.eval()
        outputs = self.model(input_ids)

        _, y_hat = outputs[0].max(1)
        predicted_idx = str(y_hat.item())
        return [predicted_idx]

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
