import json
import logging
import os
import time
from pathlib import Path

import torch
from generate import (
    Transformer,
    _load_model,
    decode_one_token,
    encode_tokens,
    logits_to_probs,
    maybe_init_dist,
    model_forward,
    multinomial_sample_one_no_sync,
    prefill,
)
from sentencepiece import SentencePieceProcessor

from ts.handler_utils.timer import timed
from ts.handler_utils.utils import send_intermediate_predict_response
from ts.torch_handler.base_handler import BaseHandler

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
        self.device = torch.device("cpu")
        self.prompt_length = 0
        self.local_rank = 0
        self.stream = False
        self.is_speculative = False
        self.draft_model = None
        self.speculate_k = 0

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
        max_seq_length = (
            max_seq_length + self.speculate_k + 1
            if self.is_speculative
            else max_seq_length
        )

        dtype = prompt.dtype
        with torch.device(self.device):
            self.model.setup_caches(max_batch_size=1, max_seq_length=max_seq_length)
            if self.is_speculative and self.draft_model is not self.model:
                self.draft_model.setup_caches(
                    max_batch_size=1, max_seq_length=max_seq_length
                )

        # create an empty tensor of the expected final shape and fill in the current tokens
        empty = torch.empty(T_new, dtype=dtype, device=self.device)
        empty[:T] = prompt
        seq = empty
        input_pos = torch.arange(0, T, device=self.device)

        next_token = self.prefill(
            self.model, prompt.view(1, -1), input_pos, **sampling_kwargs
        )
        if self.is_speculative:
            self.prefill(
                self.draft_model, prompt.view(1, -1), input_pos, **sampling_kwargs
            )

        period_id = self.tokenizer.encode(".")[0]
        text = self.tokenizer.decode([period_id] + next_token.tolist())[1:]
        if self.stream:
            send_intermediate_predict_response(
                [text],
                self.context.request_ids,
                "Intermediate Prediction success",
                200,
                self.context,
            )

        seq[T] = next_token

        input_pos = torch.tensor([T], device=self.device, dtype=torch.int)
        accept_counts = [0] * (self.speculate_k + 1)

        if self.is_speculative:
            input_pos = (
                input_pos.item()
            )  # for speculative decoding easier to keep on host
            while input_pos < T_new - 1:
                cur_token = next_token.view(())

                next_tokens = self.speculative_decode(
                    self.model,
                    self.draft_model,
                    cur_token,
                    input_pos,
                    self.speculate_k,
                    **sampling_kwargs,
                )

                accept_counts[len(next_tokens) - 1] += 1
                num_added = min(T_new - input_pos - 1, len(next_tokens))
                seq[input_pos + 1 : input_pos + num_added + 1] = next_tokens[:num_added]
                for i in next_tokens[:num_added,]:
                    callback(i)
                input_pos = input_pos + num_added
                next_token = next_tokens[-1]
        else:
            generated_tokens, _ = self.decode_n_tokens(
                next_token.view(1, -1),
                input_pos,
                max_new_tokens - 1,
                callback=callback,
                **sampling_kwargs,
            )
            seq[T + 1 :] = torch.cat(generated_tokens)

        generate_stats = {"accept_counts": accept_counts}

        return seq, generate_stats

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

    def speculative_decode(
        self,
        model: Transformer,
        draft_model: Transformer,
        cur_token: torch.Tensor,
        input_pos: int,
        speculate_k: int,
        **sampling_kwargs,
    ) -> torch.Tensor:
        # draft model inference sequentially
        device = cur_token.device
        orig_input_pos = torch.tensor(
            [input_pos], dtype=torch.int64, device=cur_token.device
        )
        draft_tokens, draft_probs = self.decode_n_tokens(
            cur_token.view(1, -1),
            orig_input_pos.clone(),
            speculate_k,
            **sampling_kwargs,
        )

        draft_tokens = torch.cat(draft_tokens)
        # parallel inference on target model using draft tokens
        target_logits = model_forward(
            model,
            torch.cat([cur_token.view(1), draft_tokens]).view(1, -1),
            torch.arange(
                input_pos, input_pos + speculate_k + 1, device=cur_token.device
            ),
        )
        target_probs = logits_to_probs(target_logits[0], **sampling_kwargs)
        draft_probs = torch.stack(draft_probs)
        # q: target prob, p: draft prob
        # q >= p: always accept draft token
        # q < p: q/p prob to accept draft token
        p = draft_probs[torch.arange(0, speculate_k, device=device), draft_tokens]
        q = target_probs[torch.arange(0, speculate_k, device=device), draft_tokens]
        accept_draft_prob = torch.minimum(torch.ones(()), q[:speculate_k] / p)
        rejected_locations = (
            torch.rand_like(accept_draft_prob) > accept_draft_prob
        ).nonzero()

        if rejected_locations.shape[0] == 0:  # All draft tokens have been accepted
            accept_length = speculate_k + 1
            last_token = multinomial_sample_one_no_sync(target_probs[-1])
            # fill last token into draft model
            model_forward(
                draft_model,
                draft_tokens[-1].view(1, -1),
                orig_input_pos + speculate_k,
            )
            return torch.cat([draft_tokens, last_token])
        else:
            accept_length = rejected_locations[0].item()
            p = draft_probs[accept_length]
            q = target_probs[accept_length]
            new = q - p
            new = torch.where(new > 0, new, 0.0)
            new = new / new.sum()
            next_token = multinomial_sample_one_no_sync(new)
            return torch.cat([draft_tokens[:accept_length], next_token])
