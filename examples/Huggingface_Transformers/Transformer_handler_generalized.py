from abc import ABC
import json
import logging
import os
import ast
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForQuestionAnswering,AutoModelForTokenClassification

from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)


class TransformersSeqClassifierHandler(BaseHandler, ABC):
    """
    Transformers sequence classifier handler class. This handler takes a text (string) and
    as input and returns the classification text based on the serialized transformers checkpoint.
    """
    def __init__(self):
        super(TransformersSeqClassifierHandler, self).__init__()
        self.initialized = False

    def initialize(self, ctx):
        self.manifest = ctx.manifest

        properties = ctx.system_properties
        model_dir = properties.get("model_dir")
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")
        #read configs
        options_file_path = os.path.join(model_dir, "setup_config.json")

        if os.path.isfile(options_file_path):
            with open(options_file_path) as g:
                self.setup_config = json.load(g)
        else:
            logger.warning('Missing the options.json file. Inference output will not include class name.')

        #Loading the model and tokenizer from checkpoint and config files based on the user's choice of mode
        #further setup config can be added.

        if self.setup_config["mode"]== "classification":
            self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
            if not os.path.isfile(os.path.join(model_dir, "vocab.txt")):
                self.tokenizer = AutoTokenizer.from_pretrained(self.setup_config["model_name"],do_lower_case=self.setup_config["do_lower_case"])
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(model_dir,do_lower_case=self.setup_config["do_lower_case"])
        elif self.setup_config["mode"]== "question_answer":
            self.model = AutoModelForQuestionAnswering.from_pretrained(model_dir)
            if not os.path.isfile(os.path.join(model_dir, "vocab.txt")):
                self.tokenizer = AutoTokenizer.from_pretrained(self.setup_config["model_name"],do_lower_case=self.setup_config["do_lower_case"])
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(model_dir,do_lower_case=self.setup_config["do_lower_case"])
        elif self.setup_config["mode"]== "token_classification":
            self.model = AutoModelForTokenClassification.from_pretrained(model_dir)
            if not os.path.isfile(os.path.join(model_dir, "vocab.txt")):
                self.tokenizer = AutoTokenizer.from_pretrained(self.setup_config["model_name"],do_lower_case=self.setup_config["do_lower_case"])
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(model_dir,do_lower_case=self.setup_config["do_lower_case"])

        self.model.to(self.device)
        self.model.eval()

        logger.debug('Transformer model from path {0} loaded successfully'.format(model_dir))

        # Read the mapping file, index to object name
        mapping_file_path = os.path.join(model_dir, "index_to_name.json")

        if os.path.isfile(mapping_file_path):
            with open(mapping_file_path) as f:
                self.mapping = json.load(f)
        else:
            logger.warning('Missing the index_to_name.json file. Inference output will not include class name.')

        self.initialized = True

    def preprocess(self, data):
        """ Basic text preprocessing, based on the user's chocie of application mode.
        """
        text = data[0].get("data")
        if text is None:
            text = data[0].get("body")
        input_text = text.decode('utf-8')
        logger.info("Received text: '%s'", input_text)

        if self.setup_config["mode"]== "classification" or self.setup_config["mode"]== "token_classification" :
            inputs = self.tokenizer.encode_plus(input_text, add_special_tokens = True, return_tensors = 'pt')
        elif self.setup_config["mode"]== "question_answer":
            # the sample text for question_answer should be formated as dictionary
            # with question and text as keys and related text as values.
            # we use this format here seperate question and text for encoding.
            received_json= ast.literal_eval(input_text)
            question = received_json["question"]
            text = received_json["text"]
            inputs = self.tokenizer.encode_plus(question, text, add_special_tokens=True, return_tensors="pt")

        return inputs

    def inference(self, inputs):
        """ Predict the class (or classes) of the received text using the serialized transformers checkpoint.
        """

        for key in inputs.keys():
            inputs[key]= inputs[key].to(self.device)

        if self.setup_config["mode"]== "classification":
            input_ids = inputs["input_ids"]
            predictions = self.model(input_ids)
            prediction = predictions[0].argmax(1).item()

            logger.info("Model predicted: '%s'", prediction)

            if self.mapping:
                prediction = self.mapping[str(prediction)]

        elif self.setup_config["mode"]== "question_answer":
            # the output should be only answer_start and answer_end
            # we are outputing the words just for demonstration.
            answer_start_scores, answer_end_scores = self.model(**inputs)
            answer_start = torch.argmax(answer_start_scores)  # Get the most likely beginning of answer with the argmax of the score
            answer_end = torch.argmax(answer_end_scores) + 1  # Get the most likely end of answer with the argmax of the score
            input_ids = inputs["input_ids"].tolist()[0]
            prediction = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))

            logger.info("Model predicted: '%s'", prediction)

        elif self.setup_config["mode"]== "token_classification":

            outputs = self.model(**inputs)[0]
            predictions = torch.argmax(outputs, dim=2)
            tokens = self.tokenizer.tokenize(self.tokenizer.decode(inputs["input_ids"][0]))
            if self.mapping:
                label_list = self.mapping["label_list"]
            label_list = label_list.strip('][').split(', ')
            prediction = [(token, label_list[prediction]) for token, prediction in zip(tokens, predictions[0].tolist())]

            logger.info("Model predicted: '%s'", prediction)

        return [prediction]

    def postprocess(self, inference_output):
        # TODO: Add any needed post-processing of the model predictions here
        return inference_output


_service = TransformersSeqClassifierHandler()


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
