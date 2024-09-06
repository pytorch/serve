import json
import logging
import time

import torch
from tensorrt_llm.runtime import ModelRunner
from transformers import AutoTokenizer

from ts.handler_utils.utils import send_intermediate_predict_response
from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)


class TRTLLMHandler(BaseHandler):
    def __init__(self):
        super().__init__()

        self.trt_llm_engine = None
        self.tokenizer = None
        self.model = None
        self.model_dir = None
        self.lora_ids = {}
        self.adapters = None
        self.initialized = False

    def initialize(self, ctx):
        self.model_dir = ctx.system_properties.get("model_dir")

        trt_llm_engine_config = ctx.model_yaml_config.get("handler").get(
            "trt_llm_engine_config"
        )

        tokenizer_dir = ctx.model_yaml_config.get("handler").get("tokenizer_dir")
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_dir,
            legacy=False,
            padding_side="left",
            truncation_side="left",
            trust_remote_code=True,
            use_fast=True,
        )

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.trt_llm_engine = ModelRunner.from_dir(**trt_llm_engine_config)
        self.initialized = True

    async def handle(self, data, context):
        start_time = time.time()

        metrics = context.metrics

        data_preprocess = await self.preprocess(data)
        output, input_batch = await self.inference(data_preprocess, context)
        output = await self.postprocess(output, input_batch, context)

        stop_time = time.time()
        metrics.add_time(
            "HandlerTime", round((stop_time - start_time) * 1000, 2), None, "ms"
        )
        return output

    async def preprocess(self, requests):
        input_batch = []
        assert len(requests) == 1, "Expecting batch_size = 1"
        for req_data in requests:
            data = req_data.get("data") or req_data.get("body")
            if isinstance(data, (bytes, bytearray)):
                data = data.decode("utf-8")

            prompt = data.get("prompt")
            temperature = data.get("temperature", 1.0)
            max_new_tokens = data.get("max_new_tokens", 50)
            input_ids = self.tokenizer.encode(
                prompt, add_special_tokens=True, truncation=True
            )
            input_batch.append(input_ids)

        input_batch = [torch.tensor(x, dtype=torch.int32) for x in input_batch]

        return (input_batch, temperature, max_new_tokens)

    async def inference(self, input_batch, context):
        input_ids_batch, temperature, max_new_tokens = input_batch

        with torch.no_grad():
            outputs = self.trt_llm_engine.generate(
                batch_input_ids=input_ids_batch,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                end_id=self.tokenizer.eos_token_id,
                pad_id=self.tokenizer.pad_token_id,
                output_sequence_lengths=True,
                streaming=True,
                return_dict=True,
            )
        return outputs, input_ids_batch

    async def postprocess(self, inference_outputs, input_batch, context):
        for inference_output in inference_outputs:
            output_ids = inference_output["output_ids"]
            sequence_lengths = inference_output["sequence_lengths"]

            batch_size, _, _ = output_ids.size()
            for batch_idx in range(batch_size):
                output_end = sequence_lengths[batch_idx][0]
                outputs = output_ids[batch_idx][0][output_end - 1 : output_end].tolist()
                output_text = self.tokenizer.decode(outputs)
                send_intermediate_predict_response(
                    [json.dumps({"text": output_text})],
                    context.request_ids,
                    "Result",
                    200,
                    context,
                )
        return [""] * len(input_batch)
