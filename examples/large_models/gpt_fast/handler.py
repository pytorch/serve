import json
import logging
import time
from pathlib import Path

import torch
from generate import _load_model, decode_one_token, encode_tokens, prefill
from sentencepiece import SentencePieceProcessor

from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)


class GptHandler(BaseHandler):
    def __init__(self):
        super().__init__()

        self.model = None
        self.tokenizer = None
        self.prefill = prefill
        self.decode_one_token = decode_one_token
        self.initialized = False
        self.device = torch.device("cpu")

    def initialize(self, ctx):
        properties = ctx.system_properties
        if torch.cuda.is_available() and properties.get("gpu_id") is not None:
            self.map_location = "cuda"
            self.device = torch.device(
                self.map_location + ":" + str(properties.get("gpu_id"))
            )

        checkpoint_path = Path(ctx.model_yaml_config["handler"]["converted_ckpt_dir"])
        assert checkpoint_path.is_file(), checkpoint_path

        tokenizer_path = checkpoint_path.parent / "tokenizer.model"
        assert tokenizer_path.is_file(), tokenizer_path

        logger.info("Loading model ...")
        t0 = time.time()
        self.model = _load_model(checkpoint_path, self.device, torch.bfloat16, False)
        torch.cuda.synchronize()
        logger.info(f"Time to load model: {time.time() - t0:.02f} seconds")

        self.tokenizer = SentencePieceProcessor(model_file=str(tokenizer_path))

        if ctx.model_yaml_config["handler"]["compile"]:
            self.decode_one_token = torch.compile(
                self.decode_one_token, mode="reduce-overhead", fullgraph=True
            )
            self.prefill = torch.compile(self.prefill, fullgraph=True, dynamic=True)

        torch.manual_seed(42 * 42)

        self.initialized = True

    def preprocess(self, requests):
        req_data = requests[0]

        input_data = req_data.get("data") or req_data.get("body")

        if isinstance(input_data, (bytes, bytearray)):
            input_data = input_data.decode("utf-8")

        input_data = json.loads(input_data)

        prompt = input_data["prompt"]

        encoded = encode_tokens(self.tokenizer, prompt, bos=True, device=self.device)

        return {
            "encoded": encoded,
            "max_new_tokens": input_data.get("max_new_tokens", 50),
        }

    def inference(self, input_data):
        y = self.generate(
            input_data["encoded"],
            input_data["max_new_tokens"],
            callback=lambda x: x,
            temperature=0.8,
            top_k=1,
        )
        return y

    def postprocess(self, y):
        return [self.tokenizer.decode(y.tolist())]

    @torch.no_grad()
    def generate(
        self,
        prompt: torch.Tensor,
        max_new_tokens: int,
        *,
        callback=lambda x: x,
        **sampling_kwargs,
    ) -> torch.Tensor:
        """
        Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.
        """
        # create an empty tensor of the expected final shape and fill in the current tokens
        T = prompt.size(0)
        T_new = T + max_new_tokens

        max_seq_length = min(T_new, self.model.config.block_size)

        device, dtype = prompt.device, prompt.dtype
        with torch.device(device):
            self.model.setup_caches(max_batch_size=1, max_seq_length=max_seq_length)

        # create an empty tensor of the expected final shape and fill in the current tokens
        empty = torch.empty(T_new, dtype=dtype, device=device)
        empty[:T] = prompt
        seq = empty
        input_pos = torch.arange(0, T, device=device)

        next_token = self.prefill(
            self.model, prompt.view(1, -1), input_pos, **sampling_kwargs
        )
        seq[T] = next_token

        input_pos = torch.tensor([T], device=device, dtype=torch.int)

        generated_tokens, _ = self.decode_n_tokens(
            next_token.view(1, -1),
            input_pos,
            max_new_tokens - 1,
            callback=callback,
            **sampling_kwargs,
        )
        seq[T + 1 :] = torch.cat(generated_tokens)

        return seq

    def decode_n_tokens(
        self,
        cur_token: torch.Tensor,
        input_pos: torch.Tensor,
        num_new_tokens: int,
        callback=lambda _: _,
        **sampling_kwargs,
    ):
        new_tokens, new_probs = [], []
        for i in range(num_new_tokens):
            with torch.backends.cuda.sdp_kernel(
                enable_flash=False, enable_mem_efficient=False, enable_math=True
            ):  # Actually better for Inductor to codegen attention here
                next_token, next_prob = self.decode_one_token(
                    self.model, cur_token, input_pos, **sampling_kwargs
                )
            input_pos += 1
            new_tokens.append(next_token.clone())
            callback(new_tokens[-1])
            new_probs.append(next_prob.clone())
            cur_token = next_token.view(1, -1)
        return new_tokens, new_probs
