import logging
import pathlib

from vllm import EngineArgs, LLMEngine, SamplingParams
from vllm.lora.request import LoRARequest

from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)


class BaseVLLMHandler(BaseHandler):
    def __init__(self):
        super().__init__()

        self.vllm_engine = None
        self.model = None
        self.model_dir = None
        self.lora_ids = {}
        self.adapters = None
        self.initialized = False

    def initialize(self, ctx):
        ctx.cache = {}

        self.model_dir = ctx.system_properties.get("model_dir")
        vllm_engine_config = self._get_vllm_engine_config(
            ctx.model_yaml_config.get("handler", {})
        )
        self.adapters = ctx.model_yaml_config.get("handler", {}).get("adapters", {})
        self.vllm_engine = LLMEngine.from_engine_args(vllm_engine_config)
        self.initialized = True

    def preprocess(self, requests):
        for req_id, req_data in zip(self.context.request_ids.values(), requests):
            if req_id not in self.context.cache:
                data = req_data.get("data") or req_data.get("body")
                if isinstance(data, (bytes, bytearray)):
                    data = data.decode("utf-8")

                prompt = data.get("prompt")
                sampling_params = self._get_sampling_params(req_data)
                lora_request = self._get_lora_request(req_data)
                self.context.cache[req_id] = {
                    "text_len": 0,
                    "stopping_criteria": self._create_stopping_criteria(req_id),
                }
                self.vllm_engine.add_request(
                    req_id, prompt, sampling_params, lora_request=lora_request
                )

        return requests

    def inference(self, input_batch):
        inference_outputs = self.vllm_engine.step()
        results = {}

        for output in inference_outputs:
            req_id = output.request_id
            results[req_id] = {
                "text": output.outputs[0].text[
                    self.context.cache[req_id]["text_len"] :
                ],
                "tokens": output.outputs[0].token_ids[-1],
                "finished": output.finished,
            }
            self.context.cache[req_id]["text_len"] = len(output.outputs[0].text)

        return [results[i] for i in self.context.request_ids.values()]

    def postprocess(self, inference_outputs):
        self.context.stopping_criteria = [
            self.context.cache[req_id]["stopping_criteria"]
            for req_id in self.context.request_ids.values()
        ]

        return inference_outputs

    def _get_vllm_engine_config(self, handler_config: dict):
        vllm_engine_params = handler_config.get("vllm_engine_config", {})
        model = vllm_engine_params.get("model", {})
        if len(model) == 0:
            model_path = handler_config.get("model_path", {})
            assert (
                len(model_path) > 0
            ), "please define model in vllm_engine_config or model_path in handler"
            model = str(pathlib.Path(self.model_dir).joinpath(model_path))
        logger.info(f"EngineArgs model={model}")
        vllm_engine_config = EngineArgs(model=model)
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
            logger.info(f"adapter_path=${adapter_path}")
            return LoRARequest(adapter_name, lora_id, adapter_path)

        return None

    def _clean_up(self, req_id):
        del self.context.cache[req_id]

    def _create_stopping_criteria(self, req_id):
        class StoppingCriteria(object):
            def __init__(self, outer, req_id):
                self.req_id = req_id
                self.outer = outer

            def __call__(self, res):
                if res["finished"]:
                    self.outer._clean_up(self.req_id)
                return res["finished"]

        return StoppingCriteria(outer=self, req_id=req_id)

    def _set_attr_value(self, obj, config: dict):
        items = vars(obj)
        for k, v in config.items():
            if k in items:
                setattr(obj, k, v)
