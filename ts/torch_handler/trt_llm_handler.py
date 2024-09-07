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
        output, data = await self.inference(data_preprocess, context)
        output = await self.postprocess(output, data, context)

        stop_time = time.time()
        metrics.add_time(
            "HandlerTime", round((stop_time - start_time) * 1000, 2), None, "ms"
        )
        return output

    async def preprocess(self, requests):
        assert len(requests) == 1, "Expecting batch_size = 1"
        req_data = requests[0]
        data = req_data.get("data") or req_data.get("body")
        if isinstance(data, (bytes, bytearray)):
            data = data.decode("utf-8")

        prompt = data.get("prompt")
        input_ids = self.tokenizer.encode(
            prompt, add_special_tokens=True, truncation=True
        )
        del data["prompt"]
        batch_input_ids = torch.tensor([input_ids], dtype=torch.int32)
        data.update({"batch_input_ids": batch_input_ids})

        return data

    async def inference(self, data, context):
        generate_kwargs = {
            "end_id": self.tokenizer.eos_token_id,
            "pad_id": self.tokenizer.pad_token_id,
            "output_sequence_lengths": True,
            "return_dict": True,
        }
        generate_kwargs.update(data)

        with torch.no_grad():
            outputs = self.trt_llm_engine.generate(**generate_kwargs)
        return outputs, data

    async def postprocess(self, inference_outputs, data, context):
        if not data.get("streaming", False):
            output_ids = inference_outputs["output_ids"]
            sequence_lengths = inference_outputs["sequence_lengths"]

            output_end = sequence_lengths[0]
            outputs = output_ids[0][0].tolist()
            output_text = self.tokenizer.decode(outputs)
            return [output_text]

        # Streaming output
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
