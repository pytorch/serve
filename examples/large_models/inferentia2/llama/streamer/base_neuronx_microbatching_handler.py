import logging
import os
import pathlib
from threading import Thread

import torch_neuronx
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers_neuronx.config import GenerationConfig, NeuronConfig
from transformers_neuronx.constants import GQA
from transformers_neuronx.generation_utils import HuggingFaceGenerationModelAdapter
from transformers_neuronx.module import save_pretrained_split

from ts.context import Context
from ts.handler_utils.hf_batch_streamer import TextIteratorStreamerBatch
from ts.handler_utils.micro_batching import MicroBatching
from ts.handler_utils.utils import import_class, send_intermediate_predict_response
from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)


class BaseNeuronXContinuousBatchingHandler(BaseHandler):
    def __init__(self):
        super().__init__()

        self.max_new_tokens = 25
        self.max_length = 100
        self.tokenizer = None
        self.model_class = None
        self.tokenizer_class = None
        self.output_streamer = None
        # enable micro batching
        self.handle = MicroBatching(self)

    def initialize(self, ctx: Context):
        ctx.cache = {}
        model_dir = ctx.system_properties.get("model_dir")
        handler_config = ctx.model_yaml_config.get("handler", {})

        # micro batching initialization
        micro_batch_config = ctx.model_yaml_config.get("micro_batching", {})
        micro_batching_parallelism = micro_batch_config.get("parallelism", None)
        if micro_batching_parallelism:
            logger.info(
                f"Setting micro batching parallelism  from model_config_yaml: {micro_batching_parallelism}"
            )
            self.handle.parallelism = micro_batching_parallelism

        micro_batch_size = micro_batch_config.get("micro_batch_size", 1)
        logger.info(f"Setting micro batching size: {micro_batch_size}")

        self.handle.micro_batch_size = micro_batch_size

        model_checkpoint_dir = handler_config.get("model_checkpoint_dir", "")

        model_checkpoint_path = pathlib.Path(model_dir).joinpath(model_checkpoint_dir)
        model_path = pathlib.Path(model_dir).joinpath(
            handler_config.get("model_path", "")
        )

        if not model_checkpoint_path.exists():
            # Load and save the CPU model
            model_cpu = AutoModelForCausalLM.from_pretrained(
                str(model_path), low_cpu_mem_usage=True
            )
            save_pretrained_split(model_cpu, model_checkpoint_path)
            # Load and save tokenizer for the model
            tokenizer = AutoTokenizer.from_pretrained(
                str(model_path), return_tensors="pt", padding_side="left"
            )
            tokenizer.save_pretrained(model_checkpoint_path)

        os.environ["NEURONX_CACHE"] = "on"
        os.environ["NEURON_COMPILE_CACHE_URL"] = f"{model_dir}/neuron_cache"
        os.environ["NEURON_CC_FLAGS"] = handler_config.get(
            "neuron_cc_flag",
            "-O1 --model-type=transformer --enable-mixed-precision-accumulation --enable-saturate-infinity",
        )

        self.max_length = int(handler_config.get("max_length", self.max_length))
        self.max_new_tokens = int(
            handler_config.get("max_new_tokens", self.max_new_tokens)
        )

        # settings for model compilation and loading
        amp = handler_config.get("amp", "fp32")
        tp_degree = handler_config.get("tp_degree", 6)
        n_positions = handler_config.get("n_positions", [self.max_length])

        # allocate "tp_degree" number of neuron cores to the worker process
        os.environ["NEURON_RT_NUM_CORES"] = str(tp_degree)
        try:
            num_neuron_cores_available = (
                torch_neuronx.xla_impl.data_parallel.device_count()
            )
            assert num_neuron_cores_available >= int(tp_degree)
        except (RuntimeError, AssertionError) as error:
            logger.error(
                "Required number of neuron cores for tp_degree "
                + str(tp_degree)
                + " are not available: "
                + str(error)
            )

            raise error
        self._set_class(ctx)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_checkpoint_path, return_tensors="pt", padding_side="left"
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        neuron_config = NeuronConfig()
        kwargs = dict(
            tp_degree=tp_degree,
            amp=amp,
            batch_size=self.handle.micro_batch_size,
            n_positions=n_positions,
            context_length_estimate=handler_config.get(
                "context_length_estimate", [self.max_length]
            ),
            attention_layout="BSH",
            group_query_attention=GQA.REPLICATED_HEADS,
            on_device_generation=GenerationConfig(do_sample=True),
            neuron_config=neuron_config,
        )
        self.model = self.model_class.from_pretrained(model_checkpoint_path, **kwargs)
        logger.info("Starting to compile the model")
        self.model.to_neuron()
        logger.info("Model has been successfully compiled")

        model_config = AutoConfig.from_pretrained(model_checkpoint_path)
        self.model = HuggingFaceGenerationModelAdapter(model_config, self.model)
        self.output_streamer = TextIteratorStreamerBatch(
            self.tokenizer,
            batch_size=self.handle.micro_batch_size,
            skip_special_tokens=True,
        )

        logger.info("Model %s loaded successfully", ctx.model_name)
        self.initialized = True

    def preprocess(self, requests):
        inputs = []
        for req in requests:
            data = req.get("data") or req.get("body")
            if isinstance(data, (bytes, bytearray)):
                data = data.decode("utf-8")

            prompt = data.get("prompt")
            inputs.append(prompt)

        # Ensure the compiled model can handle the input received
        if len(inputs) > self.handle.micro_batch_size:
            raise ValueError(
                f"Model is compiled for batch size {self.handle.micro_batch_size} but received input of size {len(inputs)}"
            )

        # Pad input to match compiled model batch size
        inputs.extend([""] * (self.handle.micro_batch_size - len(inputs)))

        return self.tokenizer(inputs, return_tensors="pt", padding=True)

    def inference(self, inputs):
        generation_kwargs = dict(
            inputs,
            streamer=self.output_streamer,
            max_new_tokens=self.max_new_tokens,
        )
        self.model.reset_generation()
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        micro_batch_idx = self.handle.get_micro_batch_idx()
        micro_batch_req_id_map = self.get_micro_batch_req_id_map(micro_batch_idx)
        for new_text in self.output_streamer:
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

    def _set_class(self, ctx):
        handler_config = ctx.model_yaml_config.get("handler", {})
        model_class_name = handler_config.get("model_class_name", None)

        assert (
            model_class_name
        ), "model_class_name not found in the section of handler in model config yaml file"
        model_module_prefix = handler_config.get("model_module_prefix", None)
        self.model_class = import_class(
            class_name=model_class_name,
            module_prefix=model_module_prefix,
        )
