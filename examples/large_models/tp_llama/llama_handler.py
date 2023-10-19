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
        self.mode = ctx.model_yaml_config["handler"]["mode"]
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

        print(set(self.context.request_ids.keys()))

        for req_id, req_data in zip(self.context.request_ids.values(), requests):
            # Tokenizer requests which are not prefilled yet
            if not req_id in self.context.cache:
                data = req_data.get("data") or req_data.get("body")
                input_text = self.prep_input_text(data["prompt"])
                logger.debug("Received text: '%s'", input_text)

                encoded = self.tokenizer.encode(input_text, bos=True, eos=False)

                encoded = torch.tensor(encoded, dtype=torch.long, device="cuda")

                self.context.cache[req_id] = {
                    "stopping_criteria": self._create_stopping_criteria(
                        req_id,
                        max_new_tokens=min(
                            self.max_new_tokens,
                            data.get("max_new_tokens", self.max_new_tokens),
                        ),
                    ),
                    "encoded": encoded,
                    "prompt_length": encoded.size(-1),
                }
                prefill.append(req_id)
            else:
                decode.append(req_id)
        return prefill, decode

    def prep_input_text(self, input_text):
        """
        preparing a single input text using the tokenizer.
        Args:
            input_text (str): The input text to be encoded.
        Returns:
            decoded input text
        """
        if self.mode == "chat":
            try:
                return json.loads(input_text)
            except json.JSONDecodeError:
                raise ValueError(f"Invalid JSON format in text: {input_text}")

        elif self.mode == "text_completion":
            try:
                if isinstance(input_text, (bytes, bytearray)):
                    input_text = input_text.decode("utf-8")
                return input_text
            except TypeError:
                raise ValueError(
                    "Expected input_texts to contain text (string) values."
                )
        else:
            raise NotImplementedError("Unsupported mode. Please select a valid mode.")

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
            results[req_id] = self._run_prefill(req_id)

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

    def postprocess(self, inference_output):
        """Post Process Function converts the predicted response into Torchserve readable format.
        Args:
            inference_output (list): It contains the predicted response of the input text.
        Returns:
            (list): Returns a list of the Predictions and Explanations.
        """

        logger.info("Generated text: %s", inference_output)

        return [inference_output]

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

        result = {req_id: {"ids": next_token.tolist()}}

        self.context.cache[req_id]["padding"] = 0

        return result

    @torch.no_grad()
    def _run_decode(self, ids):
        assert len(ids)
        print(self.batch_idx_to_req_ids)

        encoded, padding = self._prepare_model_inputs(ids)

        logits = self.model.forward(encoded[:, -1:], encoded.size(-1), padding)

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
            results[req_id] = {"ids": next_token[idx].view(1, 1).tolist()}
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
            (bsz, max_prompt_length), pad_id, dtype=torch.long, device="cuda"
        )
        for idx, req_id in enumerate(ids):
            tokens[idx, new_padding[idx] :] = self.context.cache[req_id]["encoded"]

        for l in self.model.layers:
            for idx, (old_pad, new_pad) in enumerate(zip(old_padding, new_padding)):
                seqlen = prompt_lengths[idx]
                l.attention.cache_k[
                    idx, new_pad : new_pad + seqlen, ...
                ] = l.attention.cache_k[idx, old_pad : old_pad + seqlen, ...]
                l.attention.cache_v[
                    idx, new_pad : new_pad + seqlen, ...
                ] = l.attention.cache_v[idx, old_pad : old_pad + seqlen, ...]

        for req_id, new_pad in zip(ids, new_padding):
            self.context.cache[req_id]["padding"] = new_pad

        return tokens, torch.tensor(new_padding, dtype=torch.long, device="cuda")

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
            device="cuda",
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
            range(self.max_bsz), dtype=torch.long, device="cuda"
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
            range(self.max_bsz), dtype=torch.long, device="cuda"
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
