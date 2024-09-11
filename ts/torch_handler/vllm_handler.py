import asyncio
import logging
import os
import pathlib
import time
from unittest.mock import MagicMock

from vllm import AsyncEngineArgs, AsyncLLMEngine
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    CompletionRequest,
    ErrorResponse,
)
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
from vllm.entrypoints.openai.serving_engine import LoRAModulePath

from ts.handler_utils.utils import send_intermediate_predict_response
from ts.service import PredictionException
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
        self.chat_completion_service = None
        self.completion_service = None
        self.raw_request = None
        self.initialized = False

    def initialize(self, ctx):
        self.model_dir = ctx.system_properties.get("model_dir")
        vllm_engine_config = self._get_vllm_engine_config(
            ctx.model_yaml_config.get("handler", {})
        )

        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

        self.vllm_engine = AsyncLLMEngine.from_engine_args(vllm_engine_config)

        self.adapters = ctx.model_yaml_config.get("handler", {}).get("adapters", {})
        lora_modules = [LoRAModulePath(n, p) for n, p in self.adapters.items()]

        if vllm_engine_config.served_model_name:
            served_model_names = vllm_engine_config.served_model_name
        else:
            served_model_names = [vllm_engine_config.model]

        chat_template = ctx.model_yaml_config.get("handler", {}).get(
            "chat_template", None
        )

        loop = asyncio.get_event_loop()
        model_config = loop.run_until_complete(self.vllm_engine.get_model_config())

        self.completion_service = OpenAIServingCompletion(
            self.vllm_engine,
            model_config,
            served_model_names,
            lora_modules=lora_modules,
            prompt_adapters=None,
            request_logger=None,
        )

        self.chat_completion_service = OpenAIServingChat(
            self.vllm_engine,
            model_config,
            served_model_names,
            "assistant",
            lora_modules=lora_modules,
            prompt_adapters=None,
            request_logger=None,
            chat_template=chat_template,
        )

        async def isd():
            return False

        self.raw_request = MagicMock()
        self.raw_request.headers = {}
        self.raw_request.is_disconnected = isd

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

        return [data]

    async def inference(self, input_batch, context):
        url_path = context.get_request_header(0, "url_path")

        if url_path == "v1/models":
            models = await self.chat_completion_service.show_available_models()
            return [models.model_dump()]

        directory = {
            "v1/completions": (
                CompletionRequest,
                self.completion_service,
                "create_completion",
            ),
            "v1/chat/completions": (
                ChatCompletionRequest,
                self.chat_completion_service,
                "create_chat_completion",
            ),
        }

        RequestType, service, func = directory.get(url_path, (None, None, None))

        if RequestType is None:
            raise PredictionException(f"Unknown API endpoint: {url_path}", 404)

        request = RequestType.model_validate(input_batch[0])
        g = await getattr(service, func)(
            request,
            self.raw_request,
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
