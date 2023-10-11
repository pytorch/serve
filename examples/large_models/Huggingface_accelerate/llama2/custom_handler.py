import logging
from abc import ABC
from threading import Thread

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from ts.context import Context
from ts.handler_utils.hf_batch_streamer import TextIteratorStreamerBatch
from ts.handler_utils.micro_batching import MicroBatching
from ts.protocol.otf_message_handler import send_intermediate_predict_response
from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)
logger.info("Transformers version %s", transformers.__version__)


class LlamaHandler(BaseHandler, ABC):
    """
    Transformers handler class for sequence, token classification and question answering.
    """

    def __init__(self):
        super(LlamaHandler, self).__init__()
        self.max_length = None
        self.max_new_tokens = None
        self.tokenizer = None
        self.initialized = False
        self.output_streamer = None
        # enable micro batching
        self.handle = MicroBatching(self)

    def initialize(self, ctx: Context):
        """In this initialize function, the HF large model is loaded and
        partitioned using DeepSpeed.
        Args:
            ctx (context): It is a JSON Object containing information
            pertaining to the model artifacts parameters.
        """
        model_dir = ctx.system_properties.get("model_dir")
        self.max_length = int(ctx.model_yaml_config["handler"]["max_length"])
        self.max_new_tokens = int(ctx.model_yaml_config["handler"]["max_new_tokens"])
        model_name = ctx.model_yaml_config["handler"]["model_name"]
        model_path = f'{model_dir}/{ctx.model_yaml_config["handler"]["model_path"]}'
        seed = int(ctx.model_yaml_config["handler"]["manual_seed"])
        torch.manual_seed(seed)

        # micro batching initialization
        micro_batching_parallelism = ctx.model_yaml_config.get(
            "micro_batching", {}
        ).get("parallelism", None)
        if micro_batching_parallelism:
            logger.info(
                f"Setting micro batching parallelism  from model_config_yaml: {micro_batching_parallelism}"
            )
            self.handle.parallelism = micro_batching_parallelism

        micro_batch_size = ctx.model_yaml_config.get("micro_batching", {}).get(
            "micro_batch_size", 1
        )
        logger.info(f"Setting micro batching size: {micro_batch_size}")
        self.handle.micro_batch_size = micro_batch_size

        logger.info("Model %s loading tokenizer", ctx.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="balanced",
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            load_in_8bit=True,
            trust_remote_code=True,
        )
        if ctx.model_yaml_config["handler"]["fast_kernels"]:
            from optimum.bettertransformer import BetterTransformer

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
            batch_size=self.handle.micro_batch_size,
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
        input_text = []
        for req in requests:
            data = req.get("data") or req.get("body")
            if isinstance(data, (bytes, bytearray)):
                data = data.decode("utf-8")

            logger.info(f"received req={data}")
            input_text.append(data.strip())

        # Ensure the compiled model can handle the input received
        if len(input_text) > self.handle.micro_batch_size:
            raise ValueError(
                f"Model is compiled for batch size {self.handle.micro_batch_size} but received input of size {len(input_ids_batch)}"
            )

        # Pad input to match compiled model batch size
        input_text.extend([""] * (self.handle.micro_batch_size - len(input_text)))

        return self.tokenizer(input_text, return_tensors="pt", padding=True)

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
        generation_kwargs = dict(
            input_batch,
            max_new_tokens=self.max_new_tokens,
            streamer=self.output_streamer,
        )

        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        micro_batch_idx = self.handle.get_micro_batch_idx()
        micro_batch_req_id_map = self.get_micro_batch_req_id_map(micro_batch_idx)
        for new_text in self.output_streamer:
            logger.debug("send response stream")
            send_intermediate_predict_response(
                new_text[: len(micro_batch_req_id_map)],
                micro_batch_req_id_map,
                "Intermediate Prediction success",
                200,
                self.context,
            )

        thread.join()

        return [""] * len(micro_batch_req_id_map)

    def postprocess(self, inference_output):
        """Post Process Function converts the predicted response into Torchserve readable format.
        Args:
            inference_output (list): It contains the predicted response of the input text.
        Returns:
            (list): Returns a list of the Predictions and Explanations.
        """
        return inference_output

    def get_micro_batch_req_id_map(self, micro_batch_idx: int):
        start_idx = micro_batch_idx * self.handle.micro_batch_size
        micro_batch_req_id_map = {
            index: self.context.request_ids[batch_index]
            for index, batch_index in enumerate(
                range(start_idx, start_idx + self.handle.micro_batch_size)
            )
            if batch_index in self.context.request_ids
        }

        return micro_batch_req_id_map
