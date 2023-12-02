import json
import logging
import os
import time
from pathlib import Path
from typing import List

import torch
from generate import _load_model, decode_one_token, prefill
from sentencepiece import SentencePieceProcessor

from ts.handler_utils.timer import timed
from ts.protocol.otf_message_handler import send_intermediate_predict_response
from ts.torch_handler.base_handler import BaseHandler

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

SPECIAL_TAGS = [B_INST, E_INST, "<<SYS>>", "<</SYS>>"]
UNSAFE_ERROR = "Error: special tags are not allowed as part of the prompt."

logger = logging.getLogger(__name__)


class GptHandler(BaseHandler):
    def __init__(self):
        super().__init__()

        self.context = None
        self.model = None
        self.tokenizer = None
        self.context = None
        self.prefill = prefill
        self.decode_one_token = decode_one_token
        self.max_new_tokens = 50
        self.temperature = 0
        self.initialized = False
        self.device = torch.device("cpu")
        self.prompt_length = 0

    def initialize(self, ctx):
        self.context = ctx
        self.context.cache = {}
        properties = ctx.system_properties
        if torch.cuda.is_available():
            self.map_location = "cuda"
            self.device = torch.device(
                self.map_location + ":" + str(os.getenv("LOCAL_RANK", 0))
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

        max_seq_length = self.model.config.block_size
        max_batch_size = 2

        with torch.device(self.device):
            self.model.setup_caches(
                max_batch_size=max_batch_size, max_seq_length=max_seq_length
            )

        self.max_bsz = self.model.layers[0].attention.kv_cache.k_cache.size(0)
        self.batch_idx_to_req_ids = [
            None,
        ] * self.max_bsz

        self.tokenizer = SentencePieceProcessor(model_file=str(tokenizer_path))

        if ctx.model_yaml_config["handler"]["compile"]:
            self.decode_one_token = torch.compile(
                self.decode_one_token, mode="reduce-overhead", fullgraph=True
            )
            self.prefill = torch.compile(self.prefill, fullgraph=True, dynamic=True)

        torch.manual_seed(42 * 42)

        self.initialized = True

    # def preprocess(self, requests):
    #     for req_data in requests:

    #         input_data = req_data.get("data") or req_data.get("body")

    #         if isinstance(input_data, (bytes, bytearray)):
    #             input_data = input_data.decode("utf-8")

    #         input_data = json.loads(input_data)

    #         prompt = input_data["prompt"]

    #         encoded = encode_tokens(self.tokenizer, prompt, bos=True, device=self.device)

    #     return {
    #         "encoded": encoded,
    #         "max_new_tokens": input_data.get("max_new_tokens", 50),
    #     }

    # def inference(self, input_data):
    #     y = self.generate(
    #         input_data["encoded"],
    #         input_data["max_new_tokens"],
    #         callback=lambda x: x,
    #         temperature=0.8,
    #         top_k=1,
    #     )
    #     return y

    # def postprocess(self, y):
    #     return [self.tokenizer.decode(y.tolist())]

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
        period_id = self.tokenizer.encode(".")[0]
        text = self.tokenizer.decode([period_id] + next_token.tolist())[1:]
        send_intermediate_predict_response(
            [text],
            self.context.request_ids,
            "Intermediate Prediction success",
            200,
            self.context,
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

    def preprocess(self, requests):
        """
        Basic text preprocessing, based on the user's choice of application mode.
        Args:
            requests (list): A list of dictionaries with a "data" or "body" field, each
                            containing the input text to be processed.
        Returns:
            tuple: A tuple with two tensors: the batch of input ids and the batch of
                attention masks.
        """
        self._clean_cache()
        prefill, decode = [], []

        for req_id, req_data in zip(self.context.request_ids.values(), requests):
            # Tokenizer requests which are not prefilled yet
            if not req_id in self.context.cache:
                data = req_data.get("data") or req_data.get("body")
                input_data = self._prepare_input_data(data)

                self.context.cache[req_id] = {
                    "stopping_criteria": self._create_stopping_criteria(
                        req_id,
                        max_new_tokens=min(
                            self.max_new_tokens,
                            input_data.get("max_new_tokens", self.max_new_tokens),
                        ),
                    ),
                    "encoded": input_data["encoded"],
                    "prompt_length": input_data["encoded"].size(-1),
                    "text": input_data["prompt"],
                }
                prefill.append(req_id)
            else:
                decode.append(req_id)
        return prefill, decode

    def inference(self, *args):
        """
        Generate tokens based on prompts
        Args:
            prefil : a batch of req ids for prefill
            decode : a batch of req ids for decode
        Returns:
            list: A list of dicts with predicted values for each input text in the batch.
        """
        prefill, decode_ids = args[0]

        # Prefill requests
        results = {}
        for req_id in prefill:
            results.update(self._run_prefill(req_id))

        # Decode the rest
        decode_result = self._run_decode(decode_ids) if decode_ids else {}
        results.update(decode_result)
        return [results[i] for i in self.context.request_ids.values()]

    def postprocess(self, x):
        self.context.stopping_criteria = [
            self.context.cache[i]["stopping_criteria"]
            for i in self.context.request_ids.values()
        ]
        return x

    def _prepare_input_data(self, input_text):
        """
        preparing a single input text using the tokenizer.
        Args:
            input_text (str): The input text to be encoded.
        Returns:
            decoded input text
        """
        try:
            if isinstance(input_text, (bytes, bytearray)):
                input_text = input_text.decode("utf-8")

            input_data = json.loads(input_text)

            if input_data.get("mode", "text_completion") == "chat":
                return self._prepare_dialog(input_data)

            input_data["encoded"] = self.tokenizer.encode(input_data["prompt"])
            input_data["encoded"] = [self.tokenizer.bos_id()] + input_data["encoded"]
            input_data["encoded"] = torch.tensor(
                input_data["encoded"], dtype=torch.long, device=self.device
            )

            return input_data
        except TypeError:
            raise ValueError(
                f"Expected input_texts to contain text (string) values: {input_text}"
            )
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in text: {input_text}")

    def _prepare_dialog(self, input_data):
        dialog = input_data["dialog"]
        if dialog[0]["role"] == "system":
            dialog = [
                {
                    "role": dialog[1]["role"],
                    "content": B_SYS
                    + dialog[0]["content"]
                    + E_SYS
                    + dialog[1]["content"],
                }
            ] + dialog[2:]
        assert all([msg["role"] == "user" for msg in dialog[::2]]) and all(
            [msg["role"] == "assistant" for msg in dialog[1::2]]
        ), (
            "model only supports 'system', 'user' and 'assistant' roles, "
            "starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
        )
        dialog_tokens: List[int] = sum(
            [
                self.tokenizer.encode(
                    f"{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} ",
                    bos=True,
                    eos=True,
                )
                for prompt, answer in zip(
                    dialog[::2],
                    dialog[1::2],
                )
            ],
            [],
        )
        assert (
            dialog[-1]["role"] == "user"
        ), f"Last message must be from user, got {dialog[-1]['role']}"
        dialog_tokens += self.tokenizer.encode(
            f"{B_INST} {(dialog[-1]['content']).strip()} {E_INST}",
            bos=True,
            eos=False,
        )
        del input_data["dialog"]
        input_data["prompt"] = self.tokenizer.decode(dialog_tokens)
        input_data["encoded"] = torch.tensor(
            dialog_tokens, dtype=torch.long, device=self.device
        )
        return input_data

    @torch.no_grad()
    def _run_prefill(self, req_id):
        self._vacate_kv_cache_before_prefill()
        self.batch_idx_to_req_ids[0] = req_id

        T = self.context.cache[req_id]["encoded"].size(0)
        input_pos = torch.arange(0, T, device=self.device)

        logits = self.model.forward(
            self.context.cache[req_id]["encoded"].view(1, -1), input_pos
        )

        if self.temperature > 0:
            assert 0
            # probs = torch.softmax(logits[:, -1] / self.temperature, dim=-1)
            # next_token = sample_top_p(probs, self.top_p)
        else:
            next_token = torch.argmax(logits[:, -1], dim=-1)

        self.context.cache[req_id]["encoded"] = torch.concat(
            (self.context.cache[req_id]["encoded"], next_token.view(1)), dim=-1
        )

        current_text = self.tokenizer.decode(
            self.context.cache[req_id]["encoded"].view(-1).tolist()
        )
        prev_text_len = len(self.context.cache[req_id]["text"])
        new_text = current_text[prev_text_len:]
        self.context.cache[req_id]["text"] = current_text

        result = {req_id: {"text": new_text, "ids": next_token.view(-1).tolist()}}

        self.context.cache[req_id]["padding"] = 0

        return result

    @torch.no_grad()
    def _run_decode(self, ids):
        assert len(ids)
        encoded, padding = self._prepare_model_inputs(ids)

        input_pos = torch.tensor(
            [encoded.size(-1)], dtype=torch.long, device=self.device
        )

        logits = self.model.forward(encoded[:, -1:], input_pos - 1, padding)

        if self.temperature > 0:
            assert 0
            # probs = torch.softmax(logits[:, -1] / self.temperature, dim=-1)
            # next_token = sample_top_p(probs, self.top_p)
        else:
            next_token = torch.argmax(logits[:, -1], dim=-1)
        results = {}
        for idx, req_id in enumerate(ids):
            self.context.cache[req_id]["encoded"] = torch.concat(
                (self.context.cache[req_id]["encoded"], next_token[idx].view(1)), dim=-1
            )

            current_text = self.tokenizer.decode(
                self.context.cache[req_id]["encoded"].view(-1).tolist()
            )
            prev_text_len = len(self.context.cache[req_id]["text"])
            new_text = current_text[prev_text_len:]
            self.context.cache[req_id]["text"] = current_text

            results[req_id] = {
                "text": new_text,
                "ids": next_token[idx].view(1).tolist(),
            }

        return results

    @torch.no_grad()
    def _prepare_model_inputs(self, ids):
        self._rearrange_kv_cache_for_decode(ids)

        prompt_lengths = [
            self.context.cache[req_id]["encoded"].size(-1) for req_id in ids
        ]
        old_padding = [self.context.cache[req_id]["padding"] for req_id in ids]

        max_prompt_length = max(prompt_lengths)
        new_padding = [max_prompt_length - l for l in prompt_lengths]

        bsz = len(ids)
        pad_id = self.tokenizer.eos_id()
        tokens = torch.full(
            (bsz, max_prompt_length), pad_id, dtype=torch.long, device=self.device
        )
        for idx, req_id in enumerate(ids):
            tokens[idx, new_padding[idx] :] = self.context.cache[req_id]["encoded"]

        for l in self.model.layers:
            for idx, (old_pad, new_pad) in enumerate(zip(old_padding, new_padding)):
                seqlen = prompt_lengths[idx]
                l.attention.kv_cache.k_cache[
                    idx, new_pad : new_pad + seqlen
                ] = l.attention.kv_cache.k_cache[
                    idx, old_pad : old_pad + seqlen
                ].clone()
                l.attention.kv_cache.v_cache[
                    idx, new_pad : new_pad + seqlen
                ] = l.attention.kv_cache.v_cache[
                    idx, old_pad : old_pad + seqlen
                ].clone()
                l.attention.kv_cache.k_cache[idx, :new_pad] = 0
                l.attention.kv_cache.v_cache[idx, :new_pad] = 0

        for req_id, new_pad in zip(ids, new_padding):
            self.context.cache[req_id]["padding"] = new_pad

        return tokens, torch.tensor(new_padding, dtype=torch.long, device=self.device)

    def _rearrange_kv_cache_for_decode(self, ids: List[str]) -> None:
        req_id_to_batch_idx = {
            req_id: idx
            for idx, req_id in enumerate(self.batch_idx_to_req_ids)
            if req_id is not None
        }
        decode_indices = [req_id_to_batch_idx[req_id] for req_id in ids]

        prefill_ids = list(
            set([req_id for req_id in self.batch_idx_to_req_ids if req_id is not None])
            - set(ids)
        )
        prefill_indices = [req_id_to_batch_idx[req_id] for req_id in prefill_ids]

        none_indices = [
            idx
            for idx, req_id in enumerate(self.batch_idx_to_req_ids)
            if req_id is None
        ]

        new_arrangement = torch.tensor(
            decode_indices + prefill_indices + none_indices,
            dtype=torch.long,
            device=self.device,
        )
        for l in self.model.layers:
            l.attention.kv_cache.k_cache = l.attention.kv_cache.k_cache[
                new_arrangement, ...
            ]
            l.attention.kv_cache.v_cache = l.attention.kv_cache.v_cache[
                new_arrangement, ...
            ]
        self.batch_idx_to_req_ids = [
            self.batch_idx_to_req_ids[idx]
            for idx in decode_indices + prefill_indices + none_indices
        ]

    def _vacate_kv_cache_before_prefill(self) -> torch.Tensor:
        assert self.batch_idx_to_req_ids.count(None), "Expecting an empty slot in batch"

        new_batch_idx = self.batch_idx_to_req_ids.index(None)

        rearrangement_indices = torch.tensor(
            range(self.max_bsz), dtype=torch.long, device=self.device
        )
        rearrangement_indices[new_batch_idx], rearrangement_indices[0] = (
            rearrangement_indices[0],
            rearrangement_indices[new_batch_idx],
        )
        self.batch_idx_to_req_ids[new_batch_idx], self.batch_idx_to_req_ids[0] = (
            self.batch_idx_to_req_ids[0],
            self.batch_idx_to_req_ids[new_batch_idx],
        )

        for l in self.model.layers:
            l.attention.kv_cache.k_cache = l.attention.kv_cache.k_cache[
                rearrangement_indices, ...
            ]
            l.attention.kv_cache.v_cache = l.attention.kv_cache.v_cache[
                rearrangement_indices, ...
            ]

    def _clean_cache(self):
        new_ids = set(self.context.request_ids.values())
        self.batch_idx_to_req_ids = [
            key if key in new_ids else None for key in self.batch_idx_to_req_ids
        ]
        self._compact_kv_cache()

    def _compact_kv_cache(self):
        i = 0
        j = self.max_bsz - 1

        while i < j:
            if self.batch_idx_to_req_ids[j] is None:
                j -= 1
                continue
            elif self.batch_idx_to_req_ids[i] is None:
                self._swap_kv_cache(i, j)
                self.batch_idx_to_req_ids[i], self.batch_idx_to_req_ids[j] = (
                    self.batch_idx_to_req_ids[j],
                    self.batch_idx_to_req_ids[i],
                )
                i += 1
                j -= 1
            else:
                i += 1

    def _swap_kv_cache(self, i: int, j: int) -> None:
        rearrangement_indices = torch.tensor(
            range(self.max_bsz), dtype=torch.long, device=self.device
        )
        rearrangement_indices[i], rearrangement_indices[j] = (
            rearrangement_indices[j],
            rearrangement_indices[i],
        )

        for l in self.model.layers:
            l.attention.kv_cache.k_cache = l.attention.kv_cache.k_cache[
                rearrangement_indices, ...
            ]
            l.attention.kv_cache.v_cache = l.attention.kv_cache.v_cache[
                rearrangement_indices, ...
            ]

    def _create_stopping_criteria(self, req_id, max_new_tokens=25):
        class StoppingCriteria(object):
            def __init__(
                self,
                cache,
                req_id,
                stop_token,
                max_new_tokens,
            ):
                self.req_id = req_id
                self.cache = cache
                self.max_new_tokens = max_new_tokens
                self.stop_token = stop_token

            def __call__(self, res):
                self.max_new_tokens -= 1

                if self.max_new_tokens == 0 or res["ids"][-1] == self.stop_token:
                    self.clean_up()
                    return True
                return False

            def clean_up(self):
                del self.cache[self.req_id]

        return StoppingCriteria(
            self.context.cache,
            req_id,
            self.tokenizer.eos_id,
            max_new_tokens,
        )
