import json
import logging
import os
import time
from pathlib import Path

import torch
from generate import (
    _load_model,
    decode_one_token,
    encode_tokens,
    maybe_init_dist,
    model_forward,
    prefill,
)
from sentencepiece import SentencePieceProcessor

from ts.handler_utils.timer import timed
from ts.handler_utils.utils import send_intermediate_predict_response
from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)


class VLLMBaseHandler(BaseHandler):
    def __init__(self):
        super().__init__()

        self.vllm_engine = None
        self.model = None
        self.tokenizer = None
        self.context = None
        self.prefill = prefill
        self.decode_one_token = decode_one_token
        self.initialized = False
        self.prompt_length = 0
        self.stream = False
        self.is_speculative = False
        self.draft_model = None
        self.speculate_k = 0

    def initialize(self, ctx):
        self.context = ctx
        handler_config = ctx.model_yaml_config.get("handler", {})
        vll_engine_config = handler_config.get("vll_engine_config", {})
        set_vllm_engine_args(vll_engine_config)
        vllm_engine_args = EngineArgs()
        self.vllm_engine = LLMEngine.from_engine_args(vllm_engine_args)

    def initialize(self, ctx):
        self.context = ctx
        properties = self.context.system_properties
        gpu_id = properties.get("gpu_id")
        if gpu_id is not None and int(gpu_id) < 0:
            raise ValueError("Invalid gpu_id")
        rank = maybe_init_dist()

        self.local_rank = rank if rank is not None else int(gpu_id)

        if torch.cuda.is_available():
            self.map_location = "cuda"
            self.device = torch.device(self.map_location + ":" + str(self.local_rank))

            torch.cuda.set_device(self.local_rank)

        checkpoint_path = Path(ctx.model_yaml_config["handler"]["converted_ckpt_dir"])
        assert checkpoint_path.is_file(), checkpoint_path

        tokenizer_path = checkpoint_path.parent / "tokenizer.model"
        assert tokenizer_path.is_file(), tokenizer_path

        logger.info("Loading model ...")
        t0 = time.time()
        use_tp = rank is not None
        self.model = _load_model(checkpoint_path, self.device, torch.bfloat16, use_tp)
        torch.cuda.synchronize()
        logger.info(f"Time to load model: {time.time() - t0:.02f} seconds")

        draft_checkpoint_path = ctx.model_yaml_config["handler"].get(
            "draft_checkpoint_path", None
        )
        self.is_speculative = draft_checkpoint_path is not None
        if self.is_speculative:
            self.draft_model = _load_model(
                Path(draft_checkpoint_path), self.device, torch.bfloat16, use_tp
            )
            self.speculate_k = ctx.model_yaml_config["handler"].get("speculate_k", 8)

        self.tokenizer = SentencePieceProcessor(model_file=str(tokenizer_path))

        if ctx.model_yaml_config["handler"]["compile"]:
            if ctx.model_yaml_config["handler"].get("fx_graph_cache", False):
                os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"

            if self.is_speculative and use_tp:
                torch._inductor.config.triton.cudagraph_trees = (
                    False  # Bug with cudagraph trees in this case
                )

            if self.is_speculative:
                global model_forward
                model_forward = torch.compile(
                    model_forward, mode="reduce-overhead", fullgraph=True
                )

            self.decode_one_token = torch.compile(
                self.decode_one_token, mode="reduce-overhead", fullgraph=True
            )
            self.prefill = torch.compile(self.prefill, fullgraph=True, dynamic=True)

        torch.manual_seed(42 * 42)

        self.stream = ctx.model_yaml_config["handler"].get("stream", True)

        self.initialized = True

    @timed
    def preprocess(self, requests):
        assert (
            len(requests) == 1
        ), "GPT fast is currently only supported with batch_size=1"
        req_data = requests[0]

        input_data = req_data.get("data") or req_data.get("body")

        if isinstance(input_data, (bytes, bytearray)):
            input_data = input_data.decode("utf-8")

        if isinstance(input_data, str):
            input_data = json.loads(input_data)

        prompt = input_data["prompt"]

        encoded = encode_tokens(self.tokenizer, prompt, bos=True, device=self.device)

        self.prompt_length = encoded.size(0)

        return {
            "encoded": encoded,
            "max_new_tokens": input_data.get("max_new_tokens", 50),
        }

    @timed
    def inference(self, input_data):
        tokenizer = self.tokenizer
        period_id = tokenizer.encode(".")[0]

        def call_me(x):
            nonlocal period_id, tokenizer
            text = self.tokenizer.decode([period_id] + x.tolist())[1:]
            send_intermediate_predict_response(
                [text],
                self.context.request_ids,
                "Intermediate Prediction success",
                200,
                self.context,
            )

        y, metrics = self.generate(
            input_data["encoded"],
            input_data["max_new_tokens"],
            callback=call_me if self.local_rank == 0 and self.stream else lambda x: x,
            temperature=0.8,
            top_k=1,
        )
        logger.info(f"Num tokens = {y.size(0) - self.prompt_length}")

        if self.is_speculative:
            counts_aggregated = [sum(i) for i in zip(*[metrics["accept_counts"]])]
            acceptance_probs = [i / sum(counts_aggregated) for i in counts_aggregated]
            logger.info(f"Acceptance probs: {acceptance_probs}")
            logger.info(
                f"Mean Accepted: {sum([idx * i for idx, i in enumerate(counts_aggregated)])/sum(counts_aggregated)}"
            )

        return y

    def postprocess(self, y):
        return [
            ""
            if self.stream
            else self.tokenizer.decode(y.tolist()[self.prompt_length :])
        ]


def set_vllm_engine_args(config: dict):
    for name, value in vars(EngineArgs).items():
        if not name.startswith("__"):
            if name in config:
                setattr(EngineArgs, name, config.get(name))
