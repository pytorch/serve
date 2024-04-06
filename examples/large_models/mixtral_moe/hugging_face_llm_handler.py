import json
import logging
from threading import Thread

import torch
import transformers
from optimum.bettertransformer import BetterTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from ts.context import Context
from ts.handler_utils.hf_batch_streamer import TextIteratorStreamerBatch
from ts.protocol.otf_message_handler import send_intermediate_predict_response
from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)
logger.info("Transformers version %s", transformers.__version__)


class HFLLMHandler(BaseHandler):
    """
    Handler for HuggingFace LLM
    """

    def __init__(self):
        super(HFLLMHandler, self).__init__()
        self.max_length = None
        self.max_new_tokens = None
        self.tokenizer = None
        self.initialized = False

    def initialize(self, ctx: Context):
        """In this initialize function, the HF large model is loaded and
        partitioned using DeepSpeed.
        Args:
            ctx (context): It is a JSON Object containing information
            pertaining to the model artifacts parameters.
        """
        properties = ctx.system_properties
        gpu_id = properties.get("gpu_id")
        if gpu_id is not None and int(gpu_id) < 0:
            raise ValueError("Invalid gpu_id")

        logger.info(f"GPU id is {gpu_id}")
        self.local_rank = int(gpu_id)

        if torch.cuda.is_available():
            self.map_location = "cuda"
            self.device = torch.device(self.map_location + ":" + str(self.local_rank))

            torch.cuda.set_device(self.local_rank)

        model_dir = ctx.system_properties.get("model_dir")
        self.max_length = int(ctx.model_yaml_config["handler"]["max_length"])
        model_name = ctx.model_yaml_config["handler"]["model_name"]
        model_path = f'{model_dir}/{ctx.model_yaml_config["handler"]["model_path"]}'
        seed = int(ctx.model_yaml_config["handler"]["manual_seed"])
        torch.manual_seed(seed)

        logger.info("Model %s loading tokenizer", ctx.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="balanced",
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
            trust_remote_code=True,
        )

        try:
            self.model = BetterTransformer.transform(self.model)
        except RuntimeError as error:
            logger.warning(
                "HuggingFace Optimum is not supporting this model,for the list of supported models, please refer to this doc,https://huggingface.co/docs/optimum/bettertransformer/overview"
            )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.add_special_tokens(
            {
                "pad_token": "<PAD>",
            }
        )
        self.model.resize_token_embeddings(self.model.config.vocab_size + 1)

        self.output_streamer = TextIteratorStreamerBatch(
            self.tokenizer,
            batch_size=1,
            skip_special_tokens=True,
        )
        logger.info("Model %s loaded successfully", ctx.model_name)
        self.initialized = True

    def preprocess(self, requests):
        """
        Basic text preprocessing, based on the user's choice of application mode.
        Args:
            requests (list): A list of dictionaries with a "data" or "body" field, each
                            containing the input text to be processed.
        Returns:
            tuple: A tuple with two tensors: the batch of input ids and the batch of
                attention masks.
        """
        input_texts = [data.get("data") or data.get("body") for data in requests]
        input_texts = [
            input_text.decode("utf-8")
            if isinstance(input_text, (bytes, bytearray))
            else input_text
            for input_text in input_texts
        ]
        input_texts = [
            json.loads(input_text) if isinstance(input_text, str) else input_text
            for input_text in input_texts
        ]
        input_ids_batch, attention_mask_batch, params = [], [], []
        logger.info(f"Batch size is {len(input_texts)}")
        for input_text in input_texts:
            logger.info(input_text)
            input_ids, attention_mask = self.encode_input_text(input_text["prompt"])
            input_ids_batch.append(input_ids)
            attention_mask_batch.append(attention_mask)
            params.append(input_text["params"])
        input_ids_batch = torch.cat(input_ids_batch, dim=0).to(self.model.device)
        attention_mask_batch = torch.cat(attention_mask_batch, dim=0).to(self.device)
        return input_ids_batch, attention_mask_batch, params

    def encode_input_text(self, input_text):
        """
        Encodes a single input text using the tokenizer.
        Args:
            input_text (str): The input text to be encoded.
        Returns:
            tuple: A tuple with two tensors: the encoded input ids and the attention mask.
        """
        inputs = self.tokenizer.encode_plus(
            input_text,
            max_length=self.max_length,
            padding=True,
            add_special_tokens=True,
            return_tensors="pt",
            truncation=True,
        )
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        return input_ids, attention_mask

    def inference(self, input_batch):
        """
        Predicts the class (or classes) of the received text using the serialized transformers
        checkpoint.
        Args:
            input_batch (tuple): A tuple with two tensors: the batch of input ids and the batch
                                of attention masks, as returned by the preprocess function.
        Returns:
            list: A list of strings with the predicted values for each input text in the batch.
        """
        input_ids_batch, attention_mask_batch, params = input_batch
        logger.info(
            f'temperature: {params[0]["temperature"]}, top_p: {params[0]["top_p"]}, max_new_tokens: {params[0]["max_new_tokens"]}'
        )
        input_ids_batch = input_ids_batch.to(self.device)

        self.output_streamer = TextIteratorStreamerBatch(
            self.tokenizer,
            batch_size=len(input_ids_batch),
            skip_special_tokens=True,
        )
        generation_kwargs = dict(
            inputs=input_ids_batch,
            attention_mask=attention_mask_batch,
            streamer=self.output_streamer,
            max_new_tokens=params[0]["max_new_tokens"],
            temperature=params[0]["temperature"],
            top_p=params[0]["top_p"],
        )
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        for new_text in self.output_streamer:
            send_intermediate_predict_response(
                new_text,
                self.context.request_ids,
                "Intermediate Prediction success",
                200,
                self.context,
            )

        thread.join()
        logger.info(f"Length of response is {len(input_ids_batch)}")

        return [""] * len(input_ids_batch)

    def postprocess(self, inference_output):
        """Post Process Function converts the predicted response into Torchserve readable format.
        Args:
            inference_output (list): It contains the predicted response of the input text.
        Returns:
            (list): Returns a list of the Predictions and Explanations.
        """
        return inference_output
