import logging
import pathlib
import time
from unittest.mock import MagicMock

from vllm import AsyncEngineArgs, AsyncLLMEngine
from vllm.entrypoints.openai.protocol import CompletionRequest, ErrorResponse
from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
from vllm.entrypoints.openai.serving_engine import LoRAModulePath

from ts.handler_utils.utils import send_intermediate_predict_response
from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)


class VLLMHandler(BaseHandler):
    def __init__(self):
        super().__init__()

        self.vllm_engine = None
        self.model_name = None
        self.model_dir = None
        self.lora_ids = {}
        self.adapters = None
        self.initialized = False

    def initialize(self, ctx):
        self.model_dir = ctx.system_properties.get("model_dir")
        vllm_engine_config = self._get_vllm_engine_config(
            ctx.model_yaml_config.get("handler", {})
        )

        self.vllm_engine = AsyncLLMEngine.from_engine_args(vllm_engine_config)

        self.adapters = ctx.model_yaml_config.get("handler", {}).get("adapters", {})
        lora_modules = [LoRAModulePath(n, p) for n, p in self.adapters.items()]

        if vllm_engine_config.served_model_name:
            served_model_name = vllm_engine_config.served_model_name
        else:
            served_model_name = [vllm_engine_config.model]

        self.completion_service = OpenAIServingCompletion(
            self.vllm_engine,
            vllm_engine_config,
            served_model_name,
            lora_modules=lora_modules,
            prompt_adapters=None,
            request_logger=None,
        )

        self.initialized = True

    async def handle(self, data, context):
        start_time = time.time()

        metrics = context.metrics

        data_preprocess = await self.preprocess(data, context)
        output = await self.inference(data_preprocess, context)
        output = await self.postprocess(output)

        stop_time = time.time()
        metrics.add_time(
            "HandlerTime", round((stop_time - start_time) * 1000, 2), None, "ms"
        )
        return output

    async def preprocess(self, requests, context):
        assert len(requests) == 1, "Expecting batch_size = 1"
        req_data = requests[0]
        data = req_data.get("data") or req_data.get("body")
        if isinstance(data, (bytes, bytearray)):
            data = data.decode("utf-8")

        return [self.prepare_completion_request(data)]

    async def inference(self, input_batch, context):
        request = input_batch[0]

        raw_request = MagicMock()
        raw_request.headers = {}

        async def isd():
            return False

        raw_request.is_disconnected = isd
        g = await self.completion_service.create_completion(
            request,
            raw_request,
        )

        if isinstance(g, ErrorResponse):
            return [g.model_dump()]
        if request.stream:
            async for response in g:
                if response != "data: [DONE]\n\n":
                    send_intermediate_predict_response(
                        [response], context.request_ids, "Result", 200, context
                    )
            return [response]
        else:
            return [g.model_dump()]

    async def postprocess(self, inference_outputs):
        return inference_outputs

    def prepare_completion_request(self, request_data):
        request = CompletionRequest.model_validate(request_data)
        return request  # , sampling_params, lora_request

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
            else:
                model = model.as_posix()
        logger.debug(f"EngineArgs model: {model}")
        vllm_engine_config = AsyncEngineArgs(model=model)
        self._set_attr_value(vllm_engine_config, vllm_engine_params)
        return vllm_engine_config

    def _set_attr_value(self, obj, config: dict):
        items = vars(obj)
        for k, v in config.items():
            if k in items:
                setattr(obj, k, v)
