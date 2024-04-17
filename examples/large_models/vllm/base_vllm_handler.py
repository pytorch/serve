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
        self.lora_ids = {}
        self.context = None
        self.initialized = False

    def initialize(self, ctx):
        self.context = ctx
        vllm_engine_config = self._get_vllm_engine_config(
            ctx.model_yaml_config.get("handler", {})
        )
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
                lora_request = self._get_lora_request(req_id, req_data)
                self.context.cache[req_id] = {
                    "stopping_criteria": self._create_stopping_criteria(req_id),
                }
                self.vllm_engine.add_request(
                    req_id, prompt, sampling_params, lora_request
                )

        return requests

    def inference(self, input_batch):
        return self.vllm_engine.step()
        results = {}

        for output in inference_outputs:
            req_id = output.request_id
            results[req_id]["output"] = {
                "text": output.outputs[0].text,
                "tokens": output.outputs[0].token_ids[-1],
            }
            results[req_id]["stopping_criteria"] = self.context.cache[req_id][
                "stopping_criteria"
            ](output)

        return [results[i] for i in self.context.request_ids.values()]

    def postprocess(self, inference_outputs):
        self.context.stopping_criteria = []
        results = []
        for output in inference_outputs:
            self.context.stopping_criteria.append(output["stopping_criteria"])
            results.append(output["output"])

        return results

    def _get_vllm_engine_config(self, handler_config: dict):
        vllm_engine_params = handler_config.get("vllm_engine_config", {})
        vllm_engine_config = EngineArgs()
        self._set_attr_value(vllm_engine_config, vllm_engine_params)
        return vllm_engine_config

    def _get_sampling_params(self, req_data: dict):
        sampling_params = SamplingParams()
        self._set_attr_value(sampling_params, req_data)

        return sampling_params

    def _get_lora_request(self, req_id, req_data: dict):
        lora_request_params = req_data.get("adapter", None)

        if lora_request_params:
            lora_name = lora_request_params.get("name", None)
            lora_path = lora_request_params.get("path", None)
            if lora_name and lora_path:
                lora_id = self.lora_ids.get(lora_name, len(self.lora_ids) + 1)
                return LoRARequest(lora_name, lora_id, lora_path)
            else:
                logger.error(f"request_id={req_id} missed adapter name or path")

        return None

    def _create_stopping_criteria(self, req_id):
        class StoppingCriteria(object):
            def __init__(self, cache, req_id):
                self.req_id = req_id
                self.cache = cache

            def __call__(self, res):
                return res.finished

            def clean_up(self):
                del self.cache[self.req_id]

        return StoppingCriteria(self.context.cache, req_id)

    def _set_attr_value(self, obj, config: dict):
        items = vars(obj).items()
        for k, v in config:
            if k in items:
                setattr(obj, k, v)
            elif k == "model_path":
                model_dir = self.context.system_properties.get("model_dir")
                model_path = pathlib.Path(model_dir).joinpath(v)
                setattr(obj, "model", model_path)
