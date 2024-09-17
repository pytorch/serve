import json
import logging
import time

from tensorrt_llm.hlapi import LLM, KvCacheConfig, SamplingParams
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

        engine_dir = ctx.model_yaml_config.get("handler").get("engine_dir")
        kv_cache_cfg = ctx.model_yaml_config.get("handler").get("kv_cache_config", {})

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

        kv_cache_config = KvCacheConfig(**kv_cache_cfg)

        self.trt_llm_engine = LLM(
            model=engine_dir, tokenizer=self.tokenizer, kv_cache_config=kv_cache_config
        )
        self.initialized = True

    async def handle(self, data, context):
        start_time = time.time()

        metrics = context.metrics

        data_preprocess = await self.preprocess(data)
        output = await self.inference(data_preprocess, context)
        output = await self.postprocess(output)

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
        return data

    async def inference(self, data, context):
        generate_kwargs = {
            "end_id": self.tokenizer.eos_token_id,
            "pad_id": self.tokenizer.pad_token_id,
        }
        prompt = data.get("prompt")
        streaming = data.get("streaming", False)
        del data["prompt"]
        if "streaming" in data:
            del data["streaming"]
        generate_kwargs.update(data)
        sampling_params = SamplingParams(**generate_kwargs)

        outputs = self.trt_llm_engine.generate_async(
            prompt, streaming=streaming, sampling_params=sampling_params
        )

        async for output in outputs:
            output_text, output_ids = (
                output.outputs[0].text,
                output.outputs[0].token_ids,
            )
            if not streaming:
                return [output_text]
            else:
                output_text = self.tokenizer.decode([output_ids[-1]])
                send_intermediate_predict_response(
                    [json.dumps({"text": output_text})],
                    context.request_ids,
                    "Result",
                    200,
                    context,
                )
        return [""]

    async def postprocess(self, outputs):
        return outputs
