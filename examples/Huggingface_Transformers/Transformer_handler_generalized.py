from abc import ABC
import json
import logging
import os
import ast
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForQuestionAnswering,AutoModelForTokenClassification
from ts.torch_handler.base_handler import BaseHandler
from captum.attr import LayerIntegratedGradients
logger = logging.getLogger(__name__)

class TransformersSeqClassifierHandler(BaseHandler, ABC):
    """
    Transformers handler class for sequence, token classification and question answering.
    """
    def __init__(self):
        super(TransformersSeqClassifierHandler, self).__init__()
        self.initialized = False

    def initialize(self, ctx):
        self.manifest = ctx.manifest
        properties = ctx.system_properties
        model_dir = properties.get("model_dir")
        serialized_file = self.manifest['model']['serializedFile']
        model_pt_path = os.path.join(model_dir, serialized_file)
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")
        #read configs for the mode, model_name, etc. from setup_config.json
        setup_config_path = os.path.join(model_dir, "setup_config.json")
        if os.path.isfile(setup_config_path):
            with open(setup_config_path) as setup_config_file:
                self.setup_config = json.load(setup_config_file)
        else:
            logger.warning('Missing the setup_config.json file.')

        #Loading the model and tokenizer from checkpoint and config files based on the user's choice of mode
        #further setup config can be added.
        if self.setup_config["save_mode"] == "torchscript":
            self.model = torch.jit.load(model_pt_path)
        elif self.setup_config["save_mode"] == "pretrained":
            if self.setup_config["mode"]== "sequence_classification":
                self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
            elif self.setup_config["mode"]== "question_answering":
                self.model = AutoModelForQuestionAnswering.from_pretrained(model_dir)
            elif self.setup_config["mode"]== "token_classification":
                self.model = AutoModelForTokenClassification.from_pretrained(model_dir)
            else:
                logger.warning('Missing the operation mode.')
        else:
            logger.warning('Missing the checkpoint or state_dict.')

        if not os.path.isfile(os.path.join(model_dir, "vocab.*")):
            self.tokenizer = AutoTokenizer.from_pretrained(self.setup_config["model_name"],do_lower_case=self.setup_config["do_lower_case"])
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir,do_lower_case=self.setup_config["do_lower_case"])

        self.model.to(self.device)
        self.model.eval()

        logger.debug('Transformer model from path {0} loaded successfully'.format(model_dir))

        # Read the mapping file, index to object name
        mapping_file_path = os.path.join(model_dir, "index_to_name.json")
        # Question answering does not need the index_to_name.json file.
        if not self.setup_config["mode"]== "question_answering":
            if os.path.isfile(mapping_file_path):
                with open(mapping_file_path) as f:
                    self.mapping = json.load(f)
            else:
                logger.warning('Missing the index_to_name.json file.')

              # ------------------------------- Captum initialization ----------------------------#
        self.lig = LayerIntegratedGradients(captum_sequence_forward, self.model.bert.embeddings)
        self.initialized = True

    def preprocess(self, data):
        """ Basic text preprocessing, based on the user's chocie of application mode.
        """
        input_text = None
        inp = data[0]
        if inp is not None:
            if isinstance(inp, dict):
                name = inp.get("name")
                if name == "context":
                    input_text = inp.get("data")
                    print("Inside KFServing preprocess, ",input_text)
            else:
                input_text = inp
        
        #input_text = text.decode('utf-8')
        max_length = self.setup_config["max_length"]
        logger.info("Received text: '%s'", input_text)
        #preprocessing text for sequence_classification and token_classification.
        if self.setup_config["mode"]== "sequence_classification" or self.setup_config["mode"]== "token_classification" :
            inputs = self.tokenizer.encode_plus(input_text,max_length = int(max_length),pad_to_max_length = True, add_special_tokens = True, return_tensors = 'pt')
        #preprocessing text for question_answering.
        elif self.setup_config["mode"]== "question_answering":
            #TODO Reading the context from a pickeled file or other fromats that
            # fits the requirements of the task in hand. If this is done then need to
            # modify the following preprocessing accordingly.

            # the sample text for question_answering in the current version
            # should be formated as dictionary with question and text as keys
            # and related text as values.
            # we use this format here seperate question and text for encoding.

            question_context= ast.literal_eval(input_text)
            question = question_context["question"]
            context = question_context["context"]
            inputs = self.tokenizer.encode_plus(question, context,max_length = int(max_length),pad_to_max_length = True, add_special_tokens=True, return_tensors="pt")

        return inputs, input_text

    def inference(self, inputs):
        """ Predict the class (or classes) of the received text using the serialized transformers checkpoint.
        """

        input_ids = inputs["input_ids"].to(self.device)
        # Handling inference for sequence_classification.
        if self.setup_config["mode"]== "sequence_classification":
            predictions = self.model(input_ids)
            prediction = predictions[0].argmax(1).item()

            logger.info("Model predicted: '%s'", prediction)

            if self.mapping:
                prediction = self.mapping[str(prediction)]
        # Handling inference for question_answering.
        elif self.setup_config["mode"]== "question_answering":
            # the output should be only answer_start and answer_end
            # we are outputing the words just for demonstration.
            answer_start_scores, answer_end_scores = self.model(input_ids)
            answer_start = torch.argmax(answer_start_scores)  # Get the most likely beginning of answer with the argmax of the score
            answer_end = torch.argmax(answer_end_scores) + 1  # Get the most likely end of answer with the argmax of the score
            input_ids = inputs["input_ids"].tolist()[0]
            prediction = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))

            logger.info("Model predicted: '%s'", prediction)
        # Handling inference for token_classification.
        elif self.setup_config["mode"]== "token_classification":
            outputs = self.model(input_ids)[0]
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
    
    def get_insights(self, text):
        """
        This function calls the layer integrated gradient to get word importance
        of the input text
        """
        input_ids, ref_input_ids, attention_mask = construct_input_ref(text,  self.tokenizer, self.device)
        all_tokens = get_word_token(input_ids, self.tokenizer)
        attributions, delta = self.lig.attribute(inputs=input_ids,
                                            baselines=ref_input_ids,
                                            target=self.target,
                                            additional_forward_args=(attention_mask, 0, self.model),
                                            return_convergence_delta=True)

        attributions_sum = summarize_attributions(attributions)
        response = {}
        response["importances"] = attributions_sum.tolist()
        response["words"] = all_tokens
        return [response]
    
    def handle(self, data, context, explain = False):
        """
        Entry point for default handler
        """

        # It can be used for pre or post processing if needed as additional request
        # information is available in context
        self.context = context
        output_explain  = None
        #explain from header 

        data,input_text = self.preprocess(data)
        output = self.inference(data)
        output_explain = self.explain_handle(context, input_text)
        output = self.postprocess(output)
        return output, output_explain

# Captum helper functions
def construct_input_ref(text, tokenizer, device):
    """
    For a given text, this function creates token id, reference id and
    attention mask based on encode which is faster for captum insights
    """
    text_ids = tokenizer.encode(text, add_special_tokens=False)
    # construct input token ids
    print("text_ids", text_ids)
    print("[tokenizer.cls_token_id]",[tokenizer.cls_token_id])
    input_ids = [tokenizer.cls_token_id] + text_ids + [tokenizer.sep_token_id]
    print("input_ids",input_ids)

    input_ids = torch.tensor([input_ids], device=device)
    # construct reference token ids
    ref_input_ids = [tokenizer.cls_token_id] + [tokenizer.pad_token_id] * len(text_ids) + [tokenizer.sep_token_id]
    ref_input_ids = torch.tensor([ref_input_ids], device=device)
    # construct attention mask
    attention_mask = torch.ones_like(input_ids)
    return input_ids, ref_input_ids, attention_mask

def captum_sequence_forward(inputs, attention_mask=None, position=0, model=None):
    """
    A custom forward function to access different positions of the predictions
    """
    model.eval()
    model.zero_grad()
    pred = model(inputs, attention_mask=attention_mask)
    pred = pred[position]
    return pred

def summarize_attributions(attributions):
    """
    Summarises the attribution across multiple runs
    """
    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    return attributions

def get_word_token(input_ids, tokenizer):
    """
    constructs word tokens from token id
    """
    indices = input_ids[0].detach().tolist()
    tokens = tokenizer.convert_ids_to_tokens(indices)
    # Remove unicode space character from BPE Tokeniser
    tokens = [token.replace("Ġ", "") for token in tokens]
    return tokens

