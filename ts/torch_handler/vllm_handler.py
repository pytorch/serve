import json
import logging
import pathlib
import time

from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
from vllm.lora.request import LoRARequest

from ts.handler_utils.utils import send_intermediate_predict_response
from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)


class VLLMHandler(BaseHandler):
    def __init__(self):
        super().__init__()

        self.vllm_engine = None
        self.model = None
        self.model_dir = None
        self.lora_ids = {}
        self.adapters = None
        self.initialized = False

    def initialize(self, ctx):
        self.model_dir = ctx.system_properties.get("model_dir")
        vllm_engine_config = self._get_vllm_engine_config(
            ctx.model_yaml_config.get("handler", {})
        )
        self.adapters = ctx.model_yaml_config.get("handler", {}).get("adapters", {})

        self.vllm_engine = AsyncLLMEngine.from_engine_args(vllm_engine_config)
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
        input_batch = []
        assert len(requests) == 1, "Expecting batch_size = 1"
        for req_data in requests:
            data = req_data.get("data") or req_data.get("body")
            if isinstance(data, (bytes, bytearray)):
                data = data.decode("utf-8")

            prompt = data.get("prompt")
            sampling_params = self._get_sampling_params(data)
            lora_request = self._get_lora_request(data)
            input_batch += [(prompt, sampling_params, lora_request)]
        return input_batch

    async def inference(self, input_batch, context):
        logger.debug(f"Inputs: {input_batch[0]}")
        prompt, params, lora = input_batch[0]
        generator = self.vllm_engine.generate(
            prompt, params, context.request_ids[0], lora
        )
        text_len = 0
        async for output in generator:
            result = {
                "text": output.outputs[0].text[text_len:],
                "tokens": output.outputs[0].token_ids[-1],
            }
            text_len = len(output.outputs[0].text)
            if not output.finished:
                send_intermediate_predict_response(
                    [json.dumps(result)], context.request_ids, "Result", 200, context
                )
        return [json.dumps(result)]

    async def postprocess(self, inference_outputs):
        return inference_outputs

    def _get_vllm_engine_config(self, handler_config: dict):
        vllm_engine_params = handler_config.get("vllm_engine_config", {})
        model = vllm_engine_params.get("model", {})
        if len(model) == 0:
            model_path = handler_config.get("model_path", {})
            assert (
                len(model_path) > 0
            ), "please define model in vllm_engine_config or model_path in handler"
            model = pathlib.Path(self.model_dir).joinpath(model_path)
            if not model.exists():
                logger.debug(
                    f"Model path ({model}) does not exist locally. Trying to give without model_dir as prefix."
                )
                model = model_path
        logger.debug(f"EngineArgs model: {model}")
        vllm_engine_config = AsyncEngineArgs(model=model)
        self._set_attr_value(vllm_engine_config, vllm_engine_params)
        return vllm_engine_config

    def _get_sampling_params(self, req_data: dict):
        sampling_params = SamplingParams()
        self._set_attr_value(sampling_params, req_data)

        return sampling_params

    def _get_lora_request(self, req_data: dict):
        adapter_name = req_data.get("lora_adapter", "")

        if len(adapter_name) > 0:
            adapter_path = self.adapters.get(adapter_name, "")
            assert len(adapter_path) > 0, f"{adapter_name} misses adapter path"
            lora_id = self.lora_ids.setdefault(adapter_name, len(self.lora_ids) + 1)
            adapter_path = str(pathlib.Path(self.model_dir).joinpath(adapter_path))
            return LoRARequest(adapter_name, lora_id, adapter_path)

        return None

    def _set_attr_value(self, obj, config: dict):
        items = vars(obj)
        for k, v in config.items():
            if k in items:
                setattr(obj, k, v)
