import logging
import os
from abc import ABC
from threading import Thread

import torch_neuronx
from transformers import AutoConfig, LlamaTokenizer, TextIteratorStreamer
from transformers_neuronx.generation_utils import HuggingFaceGenerationModelAdapter
from transformers_neuronx.llama.model import LlamaForSampling

from ts.protocol.otf_message_handler import send_intermediate_predict_response
from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)


class LLMHandler(BaseHandler, ABC):
    """
    Transformers handler class for text completion streaming on Inferentia2
    """

    def __init__(self):
        super(LLMHandler, self).__init__()
        self.initialized = False

    def initialize(self, ctx):
        self.manifest = ctx.manifest
        properties = ctx.system_properties
        model_dir = properties.get("model_dir")

        # settings for model compiliation and loading
        model_name = ctx.model_yaml_config["handler"]["model_name"]
        tp_degree = ctx.model_yaml_config["handler"]["tp_degree"]
        self.max_length = ctx.model_yaml_config["handler"]["max_length"]

        # allocate "tp_degree" number of neuron cores to the worker process
        os.environ["NEURON_RT_NUM_CORES"] = str(tp_degree)
        try:
            num_neuron_cores_available = (
                torch_neuronx.xla_impl.data_parallel.device_count()
            )
            assert num_neuron_cores_available >= int(tp_degree)
        except (RuntimeError, AssertionError) as error:
            raise RuntimeError(
                "Required number of neuron cores for tp_degree "
                + str(tp_degree)
                + " are not available: "
                + str(error)
            )

        os.environ["NEURON_CC_FLAGS"] = "--model-type=transformer-inference"

        self.tokenizer = LlamaTokenizer.from_pretrained(model_name)
        self.model = LlamaForSampling.from_pretrained(
            model_dir, batch_size=1, tp_degree=tp_degree
        )
        logger.info("Starting to compile the model")
        self.model.to_neuron()
        logger.info("Model has been successfully compiled")
        model_config = AutoConfig.from_pretrained(model_dir)
        self.model = HuggingFaceGenerationModelAdapter(model_config, self.model)
        self.output_streamer = TextIteratorStreamer(self.tokenizer)

        self.initialized = True

    def preprocess(self, requests):
        assert (
            len(requests) == 1
        ), "Only batch size 1 is supported. Received input with batch size: " + str(
            len(requests)
        )
        input_text = requests[0].get("data") or requests[0].get("body")
        if isinstance(input_text, (bytes, bytearray)):
            input_text = input_text.decode("utf-8")
        assert (
            type(input_text) == str
        ), "Expected a single text prompt as input but got: " + str(type(input_text))
        return self.tokenizer(input_text, return_tensors="pt")

    def inference(self, tokenized_input):
        generation_kwargs = dict(
            tokenized_input,
            streamer=self.output_streamer,
            max_new_tokens=self.max_length,
        )
        self.model.reset_generation()
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        for new_text in self.output_streamer:
            send_intermediate_predict_response(
                [new_text],
                self.context.request_ids,
                "Intermediate Prediction success",
                200,
                self.context,
            )

        thread.join()

        return [""]

    def postprocess(self, inference_output):
        return inference_output
