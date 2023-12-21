import json
import logging
import os
import sys
import time
from abc import ABC
from typing import List

import packaging.version
import requests
import torch
import transformers

from ts.torch_handler.base_handler import BaseHandler

current_working_directory = os.getcwd()
sys.path.insert(0, current_working_directory)
from generate import sample_top_p
from llama2 import Llama

logger = logging.getLogger(__name__)
logger.info("Transformers version %s", transformers.__version__)
if packaging.version.parse(torch.__version__) >= packaging.version.parse("2.0.0"):
    logger.info("PyTorch version is 2.0.0 or greater")
else:
    logger.info(
        "PyTorch version is less than 2.0.0, initializing with meta device needs PyTorch 2.0.0 and greater"
    )


B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

SPECIAL_TAGS = [B_INST, E_INST, "<<SYS>>", "<</SYS>>"]
UNSAFE_ERROR = "Error: special tags are not allowed as part of the prompt."


class LlamaHandler(BaseHandler, ABC):
    """
    Transformers handler class for sequence, token classification and question answering.
    """

    def __init__(self):
        super(LlamaHandler, self).__init__()
        self.initialized = False

    def initialize(self, ctx):
        """
        In this initialize function, the llama model is loaded using Fairscale and
        partitioned into multiple stages each on one device using PiPPy.
        Args:
            ctx (context): It is a JSON Object containing information
            pertaining to the model artefacts parameters.
        """
        # super().initialize(ctx)
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group("nccl")

        self.context = ctx

        self.context.cache = {}

        self.manifest = ctx.manifest
        properties = ctx.system_properties
        model_dir = properties.get("model_dir")

        seed = ctx.model_yaml_config["handler"]["manual_seed"]
        self.max_new_tokens = ctx.model_yaml_config["handler"]["max_new_tokens"]
        self.temperature = ctx.model_yaml_config["handler"]["temperature"]
        self.top_p = ctx.model_yaml_config["handler"]["top_p"]

        torch.manual_seed(seed)
        logger.info("Instantiating Llama model")
        model_load_start = time.perf_counter()
        llama_model_and_tok = Llama.build(
            model_args=f'{model_dir}/{ctx.model_yaml_config["handler"]["model_args_path"]}',
            converted_ckpt_dir=f'{model_dir}/{ctx.model_yaml_config["handler"]["converted_ckpt_dir"]}',
            tokenizer_path=f'{model_dir}/{ctx.model_yaml_config["handler"]["tokenizer_path"]}',
        )
        load_time = time.perf_counter() - model_load_start
        self.model = llama_model_and_tok.model
        self.model.eval()

        self.tokenizer = llama_model_and_tok.tokenizer

        logger.info(
            f"Llama model from path {model_dir} loaded successfully in {load_time} seconds"
        )

        self.max_bsz = self.model.layers[0].attention.cache_k.size(0)
        self.batch_idx_to_req_ids = [
            None,
        ] * self.max_bsz

        self.device = next(iter(self.model.parameters())).device

        self.initialized = True

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

            input_data["encoded"] = self.tokenizer.encode(
                input_data["prompt"], bos=True, eos=False
            )
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

        logits = self.model.forward(
            self.context.cache[req_id]["encoded"].view(1, -1), 0
        )

        if self.temperature > 0:
            probs = torch.softmax(logits[:, -1] / self.temperature, dim=-1)
            next_token = sample_top_p(probs, self.top_p)
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

        logits = self.model.forward(encoded[:, -1:], encoded.size(-1) - 1, padding)

        if self.temperature > 0:
            probs = torch.softmax(logits[:, -1] / self.temperature, dim=-1)
            next_token = sample_top_p(probs, self.top_p)
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
        pad_id = self.tokenizer.eos_id
        tokens = torch.full(
            (bsz, max_prompt_length), pad_id, dtype=torch.long, device=self.device
        )
        for idx, req_id in enumerate(ids):
            tokens[idx, new_padding[idx] :] = self.context.cache[req_id]["encoded"]

        for l in self.model.layers:
            for idx, (old_pad, new_pad) in enumerate(zip(old_padding, new_padding)):
                seqlen = prompt_lengths[idx]
                l.attention.cache_k[
                    idx, new_pad : new_pad + seqlen
                ] = l.attention.cache_k[idx, old_pad : old_pad + seqlen].clone()
                l.attention.cache_v[
                    idx, new_pad : new_pad + seqlen
                ] = l.attention.cache_v[idx, old_pad : old_pad + seqlen].clone()
                l.attention.cache_k[idx, :new_pad] = 0
                l.attention.cache_v[idx, :new_pad] = 0

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
            l.attention.cache_k = l.attention.cache_k[new_arrangement, ...]
            l.attention.cache_v = l.attention.cache_v[new_arrangement, ...]
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
            l.attention.cache_k = l.attention.cache_k[rearrangement_indices, ...]
            l.attention.cache_v = l.attention.cache_v[rearrangement_indices, ...]

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
            l.attention.cache_k = l.attention.cache_k[rearrangement_indices, ...]
            l.attention.cache_v = l.attention.cache_v[rearrangement_indices, ...]

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
