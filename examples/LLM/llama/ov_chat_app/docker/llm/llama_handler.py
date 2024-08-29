import json
import logging
import time
import re
import torch

from pathlib import Path
from tp import maybe_init_dist
from ts.handler_utils.timer import timed
from ts.torch_handler.base_handler import BaseHandler
from tokenizer import get_tokenizer

from generate import (
    generate,
    _load_model,
    decode_one_token,
    encode_tokens,
    model_forward,
    prefill,
)


logger = logging.getLogger(__name__)


class GptHandler(BaseHandler):
    def __init__(self):
        super().__init__()

        self.model = None
        self.tokenizer = None
        self.context = None
        self.prefill = prefill
        self.decode_one_token = decode_one_token
        self.initialized = False
        self.device = "cpu"
        self.prompt_length = 0
        self.local_rank = 0
        self.stream = False
        self.is_speculative = False
        self.draft_model = None
        self.speculate_k = 5
        self.draft_checkpoint_path =None

    def initialize(self, ctx):
        self.context = ctx

        checkpoint_path = Path(ctx.model_yaml_config["handler"]["converted_ckpt_dir"])
        assert checkpoint_path.is_file(), checkpoint_path

        tokenizer_path = checkpoint_path.parent / "tokenizer.model"
        assert tokenizer_path.is_file(), tokenizer_path

        rank = maybe_init_dist()
        use_tp = rank is not None
        if use_tp:
            if rank != 0:
                # only print on rank 0
                print = lambda *args, **kwargs: None

        self.device = ctx.model_yaml_config["deviceType"]

        logger.info(f"Using device={self.device}")
        precision = torch.bfloat16
        self.draft_checkpoint_path = ctx.model_yaml_config["handler"].get(
            "draft_checkpoint_path", None
        )
        self.is_speculative = self.draft_checkpoint_path is not None

        logger.info("Loading model ...")
        t0 = time.time()
        self.model = _load_model(checkpoint_path, self.device, precision, use_tp)
        logger.info(f"Time to load model: {time.time() - t0:.02f} seconds")

        if self.is_speculative:
            self.draft_model = _load_model(
                Path(self.draft_checkpoint_path), self.device, precision, use_tp
            )
            self.speculate_k = ctx.model_yaml_config["handler"].get("speculate_k", 5)
        else:
            self.draft_model = None

        self.tokenizer = get_tokenizer(tokenizer_path, checkpoint_path)

        if ctx.model_yaml_config["handler"]["compile"]:
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
        input_data["encoded"] = encoded

        return input_data

    @timed
    def inference(self, input_data):

        y, metrics = generate(
                self.model,
                input_data["encoded"],
                max_new_tokens=input_data.get("max_new_tokens", 50),
                draft_model=self.draft_model,
                interactive=False,
                callback=lambda x: x,
                temperature=input_data.get("temperature", 0.8),
                top_k=input_data.get("top_k", 200),
        )


        if self.is_speculative:
            counts_aggregated = [sum(i) for i in zip(*[metrics["accept_counts"]])]
            acceptance_probs = [i / sum(counts_aggregated) for i in counts_aggregated]
            logger.info(f"Acceptance probs: {acceptance_probs}")
            logger.info(
                f"Mean Accepted: {sum([idx * i for idx, i in enumerate(counts_aggregated)])/sum(counts_aggregated)}"
            )

        return y


    def postprocess(self, y):
        period_id = self.tokenizer.encode(".")[0]
        generated_prompts = self.tokenizer.decode([period_id] + y.tolist())[1:]
        logger.info(f"Generated prompt: {generated_prompts}")
        prompts = get_prompts_string(generated_prompts)

        return [prompts]


def get_prompts_string(input_string):
    res = re.search(r'(?:E\.g\.:\s*|Example\s*)(.*?)([.\]])', input_string)
    return res.group(1)
