from abc import ABC
import json
import logging
import os
import ast

import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)


class TransformersQuestionAnswerHandler(BaseHandler, ABC):
    """
    Transformers text classifier handler class. This handler takes a text (string) and
    as input and returns the classification text based on the serialized transformers checkpoint.

    Adapted from https://github.com/pytorch/serve/blob/master/examples/Huggingface_Transformers
    """
    def __init__(self):
        super(TransformersQuestionAnswerHandler, self).__init__()
        self.initialized = False

    def initialize(self, ctx):
        self.manifest = ctx.manifest

        properties = ctx.system_properties
        model_dir = properties.get("model_dir")
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")

        # Read model serialize/pt file
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

        self.model.to(self.device)
        self.model.eval()

        logger.debug('Transformer model from path {0} loaded successfully'.format(model_dir))

        self.initialized = True

    def preprocess(self, data):
        """ Very basic preprocessing code - only tokenizes. 
            Extend with your own preprocessing steps as needed.
        """
        text = data[0].get("data")
        if text is None:
            text = data[0].get("body")
        input_text = text.decode('utf-8')
        
        logger.info("Received decoded text: '%s'", input_text)
        
        question_context= ast.literal_eval(input_text)
        question = question_context["question"]
        context = question_context["context"]
        inputs = self.tokenizer.encode_plus(question, context, add_special_tokens=True,return_tensors="pt")

        return inputs


    def inference(self, inputs):
        """
        Predict the answer for a question in the text using a trained transformer model.
        """
        #input_ids = inputs["input_ids"].to(self.device)
        input_ids = inputs["input_ids"].tolist()[0]
        answer_start_scores, answer_end_scores = self.model(**inputs)
        answer_start = torch.argmax(answer_start_scores)  # Get the most likely beginning of answer with the argmax of the score
        answer_end = torch.argmax(answer_end_scores) + 1  # Get the most likely end of answer with the argmax of the score
        input_ids = inputs["input_ids"].tolist()[0]
        prediction = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))

        logger.info("Model predicted: '%s'", prediction)
        
        return [prediction]

    def postprocess(self, inference_output):
        return inference_output
    
    def handle(self, data, context):
        try:
            if data is None:
                 return None

            data = self.preprocess(data)
            data = self.inference(data)
            data = self.postprocess(data)

            return data
        except Exception as e:
            raise e
