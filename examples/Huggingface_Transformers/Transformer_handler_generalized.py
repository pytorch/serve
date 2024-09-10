import ast
import json
import logging
import os

import torch
import transformers
from captum.attr import LayerIntegratedGradients
from transformers import (
    AutoModelForCausalLM,
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    GPT2TokenizerFast,
)

from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)
logger.info("Transformers version %s", transformers.__version__)


class TransformersSeqClassifierHandler(BaseHandler):
    """
    Transformers handler class for sequence, token classification and question answering.
    """

    def __init__(self):
        super(TransformersSeqClassifierHandler, self).__init__()
        self.setup_config = None
        self.initialized = False

    def initialize(self, ctx):
        """In this initialize function, the BERT model is loaded and
        the Layer Integrated Gradients Algorithm for Captum Explanations
        is initialized here.
        Args:
            ctx (context): It is a JSON Object containing information
            pertaining to the model artifacts parameters.
        """
        self.manifest = ctx.manifest
        self.model_yaml_config = (
            ctx.model_yaml_config
            if ctx is not None and hasattr(ctx, "model_yaml_config")
            else {}
        )
        properties = ctx.system_properties
        model_dir = properties.get("model_dir")
        serialized_file = self.manifest["model"]["serializedFile"]
        model_pt_path = os.path.join(model_dir, serialized_file)

        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id"))
            if torch.cuda.is_available() and properties.get("gpu_id") is not None
            else "cpu"
        )

        # read configs for the mode, model_name, etc. from the handler config
        self.setup_config = self.model_yaml_config.get("handler", {})
        if not self.setup_config:
            logger.warning("Missing the handler config")

        # Loading the model and tokenizer from checkpoint and config files based on the user's choice of mode
        # further setup config can be added.
        if self.setup_config["save_mode"] == "torchscript":
            self.model = torch.jit.load(model_pt_path, map_location=self.device)
        elif self.setup_config["save_mode"] == "pretrained":
            if self.setup_config["mode"] == "sequence_classification":
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    model_dir
                )
            elif self.setup_config["mode"] == "question_answering":
                self.model = AutoModelForQuestionAnswering.from_pretrained(model_dir)
            elif self.setup_config["mode"] == "token_classification":
                self.model = AutoModelForTokenClassification.from_pretrained(model_dir)
            elif self.setup_config["mode"] == "text_generation":
                self.model = AutoModelForCausalLM.from_pretrained(model_dir)
            else:
                logger.warning("Missing the operation mode.")
            # Using the Better Transformer integration to speedup the inference
            if self.setup_config["BetterTransformer"]:
                from optimum.bettertransformer import BetterTransformer

                try:
                    self.model = BetterTransformer.transform(self.model)
                except RuntimeError as error:
                    logger.warning(
                        "HuggingFace Optimum is not supporting this model,for the list of supported models, please refer to this doc,https://huggingface.co/docs/optimum/bettertransformer/overview"
                    )
            # HF GPT2 models options can be gpt2, gpt2-medium, gpt2-large, gpt2-xl
            # this basically place different model blocks on different devices,
            # https://github.com/huggingface/transformers/blob/v4.17.0/src/transformers/models/gpt2/modeling_gpt2.py#L962
            if (
                self.setup_config["model_parallel"]
                and "gpt2" in self.setup_config["model_name"]
            ):
                self.model.parallelize()
            else:
                self.model.to(self.device)

        else:
            logger.warning("Missing the checkpoint or state_dict.")

        if "gpt2" in self.setup_config["model_name"]:
            self.tokenizer = GPT2TokenizerFast.from_pretrained(
                "gpt2", pad_token="<|endoftext|>"
            )

        elif any(
            fname
            for fname in os.listdir(model_dir)
            if fname.startswith("vocab.") and os.path.isfile(fname)
        ):
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_dir, do_lower_case=self.setup_config["do_lower_case"]
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.setup_config["model_name"],
                do_lower_case=self.setup_config["do_lower_case"],
            )

        self.model.eval()

        pt2_value = self.model_yaml_config.get("pt2", {})
        if "compile" in pt2_value:
            compile_options = pt2_value["compile"]
            if compile_options["enable"] == True:
                del compile_options["enable"]

                compile_options_str = ", ".join(
                    [f"{k} {v}" for k, v in compile_options.items()]
                )
                self.model = torch.compile(
                    self.model,
                    **compile_options,
                )
                logger.info(f"Compiled model with {compile_options_str}")
        logger.info("Transformer model from path %s loaded successfully", model_dir)

        # Read the mapping file, index to object name
        mapping_file_path = os.path.join(model_dir, "index_to_name.json")
        # Question answering does not need the index_to_name.json file.
        if not (
            self.setup_config["mode"] == "question_answering"
            or self.setup_config["mode"] == "text_generation"
        ):
            if os.path.isfile(mapping_file_path):
                with open(mapping_file_path) as f:
                    self.mapping = json.load(f)
            else:
                logger.warning("Missing the index_to_name.json file.")
        self.initialized = True

    def preprocess(self, requests):
        """Basic text preprocessing, based on the user's choice of application mode.
        Args:
            requests (str): The Input data in the form of text is passed on to the preprocess
            function.
        Returns:
            list : The preprocess function returns a list of Tensor for the size of the word tokens.
        """
        input_ids_batch = None
        attention_mask_batch = None
        for idx, data in enumerate(requests):
            input_text = data.get("data")
            if input_text is None:
                input_text = data.get("body")
            if isinstance(input_text, (bytes, bytearray)):
                input_text = input_text.decode("utf-8")
            if (
                self.setup_config["captum_explanation"]
                and not self.setup_config["mode"] == "question_answering"
            ):
                input_text_target = ast.literal_eval(input_text)
                input_text = input_text_target["text"]
            max_length = self.setup_config["max_length"]
            logger.info("Received text: '%s'", input_text)
            # preprocessing text for sequence_classification, token_classification or text_generation
            if self.setup_config["mode"] in {
                "sequence_classification",
                "token_classification",
                "text_generation",
            }:
                inputs = self.tokenizer.encode_plus(
                    input_text,
                    max_length=int(max_length),
                    pad_to_max_length=True,
                    add_special_tokens=True,
                    return_tensors="pt",
                )

            # preprocessing text for question_answering.
            elif self.setup_config["mode"] == "question_answering":
                # TODO Reading the context from a pickled file or other formats that
                # fits the requirements of the task in hand. If this is done then need to
                # modify the following preprocessing accordingly.

                # the sample text for question_answering in the current version
                # should be formatted as dictionary with question and text as keys
                # and related text as values.
                # we use this format here separate question and text for encoding.

                question_context = ast.literal_eval(input_text)
                question = question_context["question"]
                context = question_context["context"]
                inputs = self.tokenizer.encode_plus(
                    question,
                    context,
                    max_length=int(max_length),
                    pad_to_max_length=True,
                    add_special_tokens=True,
                    return_tensors="pt",
                )
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)
            # making a batch out of the received requests
            # attention masks are passed for cases where input tokens are padded.
            if input_ids.shape is not None:
                if input_ids_batch is None:
                    input_ids_batch = input_ids
                    attention_mask_batch = attention_mask
                else:
                    input_ids_batch = torch.cat((input_ids_batch, input_ids), 0)
                    attention_mask_batch = torch.cat(
                        (attention_mask_batch, attention_mask), 0
                    )
        return (input_ids_batch, attention_mask_batch)

    @torch.inference_mode
    def inference(self, input_batch):
        """Predict the class (or classes) of the received text using the
        serialized transformers checkpoint.
        Args:
            input_batch (list): List of Text Tensors from the pre-process function is passed here
        Returns:
            list : It returns a list of the predicted value for the input text
        """
        input_ids_batch, attention_mask_batch = input_batch
        inferences = []
        # Handling inference for sequence_classification.
        if self.setup_config["mode"] == "sequence_classification":
            predictions = self.model(input_ids_batch, attention_mask_batch)
            print(
                "This the output size from the Seq classification model",
                predictions[0].size(),
            )
            print("This the output from the Seq classification model", predictions)

            num_rows, num_cols = predictions[0].shape
            for i in range(num_rows):
                out = predictions[0][i].unsqueeze(0)
                y_hat = out.argmax(1).item()
                predicted_idx = str(y_hat)
                inferences.append(self.mapping[predicted_idx])
        # Handling inference for question_answering.
        elif self.setup_config["mode"] == "question_answering":
            # the output should be only answer_start and answer_end
            # we are outputing the words just for demonstration.
            if self.setup_config["save_mode"] == "pretrained":
                outputs = self.model(input_ids_batch, attention_mask_batch)
                answer_start_scores = outputs.start_logits
                answer_end_scores = outputs.end_logits
            else:
                answer_start_scores, answer_end_scores = self.model(
                    input_ids_batch, attention_mask_batch
                )
            print(
                "This the output size for answer start scores from the question answering model",
                answer_start_scores.size(),
            )
            print(
                "This the output for answer start scores from the question answering model",
                answer_start_scores,
            )
            print(
                "This the output size for answer end scores from the question answering model",
                answer_end_scores.size(),
            )
            print(
                "This the output for answer end scores from the question answering model",
                answer_end_scores,
            )

            num_rows, num_cols = answer_start_scores.shape
            # inferences = []
            for i in range(num_rows):
                answer_start_scores_one_seq = answer_start_scores[i].unsqueeze(0)
                answer_start = torch.argmax(answer_start_scores_one_seq)
                answer_end_scores_one_seq = answer_end_scores[i].unsqueeze(0)
                answer_end = torch.argmax(answer_end_scores_one_seq) + 1
                prediction = self.tokenizer.convert_tokens_to_string(
                    self.tokenizer.convert_ids_to_tokens(
                        input_ids_batch[i].tolist()[answer_start:answer_end]
                    )
                )
                inferences.append(prediction)
            logger.info("Model predicted: '%s'", prediction)
        # Handling inference for token_classification.
        elif self.setup_config["mode"] == "token_classification":
            outputs = self.model(input_ids_batch, attention_mask_batch)[0]
            print(
                "This the output size from the token classification model",
                outputs.size(),
            )
            print("This the output from the token classification model", outputs)
            num_rows = outputs.shape[0]
            for i in range(num_rows):
                output = outputs[i].unsqueeze(0)
                predictions = torch.argmax(output, dim=2)
                tokens = self.tokenizer.tokenize(
                    self.tokenizer.decode(input_ids_batch[i])
                )
                if self.mapping:
                    label_list = self.mapping["label_list"]
                label_list = label_list.strip("][").split(", ")
                prediction = [
                    (token, label_list[prediction])
                    for token, prediction in zip(tokens, predictions[0].tolist())
                ]
                inferences.append(prediction)
            logger.info("Model predicted: '%s'", prediction)

        # Handling inference for text_generation.
        if self.setup_config["mode"] == "text_generation":
            if self.setup_config["model_parallel"]:
                # Need to move the first device, as the trasnformer model has been placed there
                # https://github.com/huggingface/transformers/blob/v4.17.0/src/transformers/models/gpt2/modeling_gpt2.py#L970
                input_ids_batch = input_ids_batch.to("cuda:0")
            outputs = self.model.generate(
                input_ids_batch,
                max_new_tokens=self.setup_config["max_length"],
                do_sample=True,
                top_p=0.95,
                top_k=60,
            )
            for i, x in enumerate(outputs):
                inferences.append(
                    self.tokenizer.decode(outputs[i], skip_special_tokens=True)
                )

            logger.info("Generated text: '%s'", inferences)

        print("Generated text", inferences)
        return inferences

    def postprocess(self, inference_output):
        """Post Process Function converts the predicted response into Torchserve readable format.
        Args:
            inference_output (list): It contains the predicted response of the input text.
        Returns:
            (list): Returns a list of the Predictions and Explanations.
        """
        return inference_output

    def get_insights(self, input_batch, text, target):
        """This function initialize and calls the layer integrated gradient to get word importance
        of the input text if captum explanation has been selected through setup_config
        Args:
            input_batch (int): Batches of tokens IDs of text
            text (str): The Text specified in the input request
            target (int): The Target can be set to any acceptable label under the user's discretion.
        Returns:
            (list): Returns a list of importances and words.
        """

        if self.setup_config["captum_explanation"]:
            embedding_layer = getattr(self.model, self.setup_config["embedding_name"])
            embeddings = embedding_layer.embeddings
            self.lig = LayerIntegratedGradients(captum_sequence_forward, embeddings)
        else:
            logger.warning("Captum Explanation is not chosen and will not be available")

        if isinstance(text, (bytes, bytearray)):
            text = text.decode("utf-8")
        text_target = ast.literal_eval(text)

        if not self.setup_config["mode"] == "question_answering":
            text = text_target["text"]
        self.target = text_target["target"]

        input_ids, ref_input_ids, attention_mask = construct_input_ref(
            text, self.tokenizer, self.device, self.setup_config["mode"]
        )
        all_tokens = get_word_token(input_ids, self.tokenizer)
        response = {}
        response["words"] = all_tokens
        if (
            self.setup_config["mode"] == "sequence_classification"
            or self.setup_config["mode"] == "token_classification"
        ):
            attributions, delta = self.lig.attribute(
                inputs=input_ids,
                baselines=ref_input_ids,
                target=self.target,
                additional_forward_args=(attention_mask, 0, self.model),
                return_convergence_delta=True,
            )

            attributions_sum = summarize_attributions(attributions)
            response["importances"] = attributions_sum.tolist()
            response["delta"] = delta[0].tolist()

        elif self.setup_config["mode"] == "question_answering":
            attributions_start, delta_start = self.lig.attribute(
                inputs=input_ids,
                baselines=ref_input_ids,
                target=self.target,
                additional_forward_args=(attention_mask, 0, self.model),
                return_convergence_delta=True,
            )
            attributions_end, delta_end = self.lig.attribute(
                inputs=input_ids,
                baselines=ref_input_ids,
                target=self.target,
                additional_forward_args=(attention_mask, 1, self.model),
                return_convergence_delta=True,
            )
            attributions_sum_start = summarize_attributions(attributions_start)
            attributions_sum_end = summarize_attributions(attributions_end)
            response["importances_answer_start"] = attributions_sum_start.tolist()
            response["importances_answer_end"] = attributions_sum_end.tolist()
            response["delta_start"] = delta_start[0].tolist()
            response["delta_end"] = delta_end[0].tolist()

        return [response]


def construct_input_ref(text, tokenizer, device, mode):
    """For a given text, this function creates token id, reference id and
    attention mask based on encode which is faster for captum insights
    Args:
        text (str): The text specified in the input request
        tokenizer (AutoTokenizer Class Object): To word tokenize the input text
        device (cpu or gpu): Type of the Environment the server runs on.
    Returns:
        input_id(Tensor): It attributes to the tensor of the input tokenized words
        ref_input_ids(Tensor): Ref Input IDs are used as baseline for the attributions
        attention mask() :  The attention mask is a binary tensor indicating the position
         of the padded indices so that the model does not attend to them.
    """
    if mode == "question_answering":
        question_context = ast.literal_eval(text)
        question = question_context["question"]
        context = question_context["context"]
        text_ids = tokenizer.encode(question, context, add_special_tokens=False)

    text_ids = tokenizer.encode(text, add_special_tokens=False)
    # construct input token ids
    logger.info("text_ids %s", text_ids)
    logger.info("[tokenizer.cls_token_id] %s", [tokenizer.cls_token_id])
    input_ids = [tokenizer.cls_token_id] + text_ids + [tokenizer.sep_token_id]
    logger.info("input_ids %s", input_ids)

    input_ids = torch.tensor([input_ids], device=device)
    # construct reference token ids
    ref_input_ids = (
        [tokenizer.cls_token_id]
        + [tokenizer.pad_token_id] * len(text_ids)
        + [tokenizer.sep_token_id]
    )
    ref_input_ids = torch.tensor([ref_input_ids], device=device)
    # construct attention mask
    attention_mask = torch.ones_like(input_ids)
    return input_ids, ref_input_ids, attention_mask


def captum_sequence_forward(inputs, attention_mask=None, position=0, model=None):
    """This function is used to get the predictions from the model and this function
    can be used independent of the type of the BERT Task.
    Args:
        inputs (list): Input for Predictions
        attention_mask (list, optional): The attention mask is a binary tensor indicating the position
         of the padded indices so that the model does not attend to them, it defaults to None.
        position (int, optional): Position depends on the BERT Task.
        model ([type], optional): Name of the model, it defaults to None.
    Returns:
        list: Prediction Outcome
    """
    model.eval()
    model.zero_grad()
    pred = model(inputs, attention_mask=attention_mask)
    pred = pred[position]
    return pred


def summarize_attributions(attributions):
    """Summarizes the attribution across multiple runs
    Args:
        attributions ([list): attributions from the Layer Integrated Gradients
    Returns:
        list : Returns the attributions after normalizing them.
    """
    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    return attributions


def get_word_token(input_ids, tokenizer):
    """constructs word tokens from token id using the BERT's
    Auto Tokenizer
    Args:
        input_ids (list): Input IDs from construct_input_ref method
        tokenizer (class): The Auto Tokenizer Pre-Trained model object
    Returns:
        (list): Returns the word tokens
    """
    indices = input_ids[0].detach().tolist()
    tokens = tokenizer.convert_ids_to_tokens(indices)
    # Remove unicode space character from BPE Tokeniser
    tokens = [token.replace("Ġ", "") for token in tokens]
    return tokens
