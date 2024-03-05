import logging
import os
import pathlib

import torch
import torch_neuronx
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers_neuronx.config import ContinuousBatchingConfig, NeuronConfig
from transformers_neuronx.module import save_pretrained_split
from transformers_neuronx.sampling import select_tokens

from ts.context import Context
from ts.handler_utils.utils import import_class
from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)


class BaseNeuronXContinuousBatchingHandler(BaseHandler):
    def __init__(self):
        super().__init__()

        self.batch_size = 2
        self.max_new_tokens = 25
        self.max_length = 100
        self.tokenizer = None
        self.decode_next_tokens = None
        self.decode_cache_ids = None
        self.decode_seq_ids = None
        # the queue of seq_ids which are available for a new request
        self.empty_seq_ids = []
        # map seq_id to req_id
        self.seq_id_to_req_id = {}
        self.model_class = None
        self.tokenizer_class = None

    def initialize(self, ctx: Context):
        ctx.cache = {}
        model_dir = ctx.system_properties.get("model_dir")
        handler_config = ctx.model_yaml_config.get("handler", {})
        model_checkpoint_dir = handler_config.get("model_checkpoint_dir", "")

        model_checkpoint_path = pathlib.Path(model_dir).joinpath(model_checkpoint_dir)
        model_path = pathlib.Path(model_dir).joinpath(
            handler_config.get("model_path", "")
        )

        if not model_checkpoint_path.exists():
            # Load and save the CPU model
            model_cpu = AutoModelForCausalLM.from_pretrained(
                str(model_path), low_cpu_mem_usage=True
            )
            save_pretrained_split(model_cpu, model_checkpoint_path)
            # Load and save tokenizer for the model
            tokenizer = AutoTokenizer.from_pretrained(
                str(model_path), return_tensors="pt", padding_side="left"
            )
            tokenizer.save_pretrained(model_checkpoint_path)

        os.environ["NEURONX_CACHE"] = "on"
        os.environ["NEURON_COMPILE_CACHE_URL"] = f"{model_dir}/neuron_cache"
        os.environ[
            "NEURON_CC_FLAGS"
        ] = "-O1 --model-type=transformer --enable-mixed-precision-accumulation"

        self.max_length = int(handler_config.get("max_length", self.max_length))
        self.max_new_tokens = int(
            handler_config.get("max_new_tokens", self.max_new_tokens)
        )
        self.batch_size = int(handler_config.get("batch_size", self.batch_size))

        # settings for model compilation and loading
        amp = handler_config.get("amp", "fp32")
        tp_degree = handler_config.get("tp_degree", 6)

        # allocate "tp_degree" number of neuron cores to the worker process
        os.environ["NEURON_RT_NUM_CORES"] = str(tp_degree)
        try:
            num_neuron_cores_available = (
                torch_neuronx.xla_impl.data_parallel.device_count()
            )
            assert num_neuron_cores_available >= int(tp_degree)
        except (RuntimeError, AssertionError) as error:
            logger.error(
                "Required number of neuron cores for tp_degree "
                + str(tp_degree)
                + " are not available: "
                + str(error)
            )

            raise error
        self._set_class(ctx)
        self.tokenizer = self.tokenizer_class.from_pretrained(
            model_checkpoint_path, return_tensors="pt", padding_side="left"
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        continuous_batching_config = ContinuousBatchingConfig(
            batch_size_for_shared_caches=self.batch_size
        )
        neuron_config = NeuronConfig(continuous_batching=continuous_batching_config)
        kwargs = dict(
            tp_degree=tp_degree,
            amp=amp,
            batch_size=self.batch_size,
            n_positions=[self.max_length],
            context_length_estimate=handler_config.get(
                "context_length_estimate", [self.max_length]
            ),
            neuron_config=neuron_config,
        )
        self.model = self.model_class.from_pretrained(model_checkpoint_path, **kwargs)
        logger.info("Starting to compile the model")
        self.model.to_neuron()
        logger.info("Model has been successfully compiled")

        # 1D: [seq_id]
        # an empty slot if seq_id is -1, otherwise 0
        self.decode_seq_ids = torch.full([self.batch_size], -1)
        # 2D:[batch_size, next_cache_id]
        self.decode_cache_ids = torch.zeros(self.batch_size, 1, dtype=torch.int64)
        # 2D: [batch_size, next_token]
        self.decode_next_tokens = torch.zeros(self.batch_size, 1, dtype=torch.int64)

        for seq_id in reversed(range(self.batch_size)):
            self.empty_seq_ids.append(seq_id)

        logger.info("Model %s loaded successfully", ctx.model_name)
        self.initialized = True

    def preprocess(self, requests):
        prefill_req_ids, prefill_seq_ids, prefill_input_text, req_decode_seq_ids = (
            [],
            [],
            [],
            [],
        )
        for req_id, req_data in zip(self.context.request_ids.values(), requests):
            if req_id not in self.context.cache:
                prefill_req_ids.append(req_id)
                seq_id = self._get_empty_seq_id()
                self.seq_id_to_req_id[seq_id] = req_id
                prefill_seq_ids.append(seq_id)

                data = req_data.get("data") or req_data.get("body")
                if isinstance(data, (bytes, bytearray)):
                    data = data.decode("utf-8")

                prompt = data.get("prompt")
                max_new_tokens = int(data.get("max_new_tokens", self.max_new_tokens))
                prefill_input_text.append(prompt)

                self.context.cache[req_id] = {
                    "seq_id": seq_id,
                    "stopping_criteria": self._create_stopping_criteria(
                        req_id=req_id, seq_id=seq_id, max_new_tokens=max_new_tokens
                    ),
                }
            else:
                req_decode_seq_ids.append(self.context.cache[req_id]["seq_id"])

        prefill_tokens = None
        if len(prefill_req_ids) > 0:
            prefill_tokens = self.tokenizer(
                prefill_input_text, return_tensors="pt", padding=True
            )
        return prefill_input_text, prefill_tokens, prefill_seq_ids, req_decode_seq_ids

    def inference(self, inputs):
        prefill_input_text, prefill_tokens, prefill_seq_ids, req_decode_seq_ids = inputs
        results = {}

        if len(prefill_seq_ids) > 0:
            prefill_next_tokens, prefill_cache_ids = self._run_prefill(
                prefill_tokens, prefill_seq_ids
            )
            for i, prefill_seq_id in enumerate(prefill_seq_ids):
                self._update_results(
                    results,
                    prefill_seq_id,
                    i,
                    prefill_cache_ids,
                    prefill_next_tokens,
                    prefill_tokens=prefill_tokens,
                    prefill_input_text=prefill_input_text,
                )

        if len(req_decode_seq_ids) > 0:
            local_decode_seq_ids = torch.cat(torch.where(self.decode_seq_ids > -1))
            local_decode_cache_ids = self.decode_cache_ids[local_decode_seq_ids]
            local_decode_next_tokens = self.decode_next_tokens[local_decode_seq_ids]

            local_next_tokens = self._run_decode(
                local_decode_next_tokens, local_decode_cache_ids, local_decode_seq_ids
            )

            filter_prefill_seq_ids = (
                torch.isin(local_decode_seq_ids, torch.as_tensor(prefill_seq_ids))
                if len(prefill_seq_ids) > 0
                else torch.full(local_decode_seq_ids.shape, False)
            )

            local_decode_cache_ids = local_decode_cache_ids + 1
            for i, is_prefill_seq_id in enumerate(filter_prefill_seq_ids):
                if not is_prefill_seq_id:
                    seq_id = local_decode_seq_ids[i].item()

                    if seq_id in req_decode_seq_ids:
                        self._update_results(
                            results,
                            seq_id,
                            i,
                            local_decode_cache_ids,
                            local_next_tokens,
                        )
                    else:
                        req_id = self._get_req_id(seq_id)
                        logger.warning(
                            f"Found request id:{req_id} with seq_id:{seq_id} in local_decode_seq_ids, but not in batch requests. Delete it"
                        )
                        self._clean_up(seq_id, req_id)

        return [results[i] for i in self.context.request_ids.values()]

    def postprocess(self, inference_output):
        self.context.stopping_criteria = [
            self.context.cache[req_id]["stopping_criteria"]
            for req_id in self.context.request_ids.values()
        ]

        return inference_output

    def _get_empty_seq_id(self):
        if len(self.empty_seq_ids) == 0:
            # clean up dead req_ids due to client disconnction
            self._clean_dead_reqs()

        assert len(self.empty_seq_ids) > 0
        return self.empty_seq_ids.pop()

    def _add_empty_seq_id(self, seq_id):
        self.empty_seq_ids.append(seq_id)

    def _get_seq_id(self, req_id):
        seq_id = None
        cache = self.context.cache.get(req_id, None)
        if cache:
            seq_id = cache["seq_id"]
        assert seq_id is not None, "{req_id} must have seq_id"
        return seq_id

    def _get_req_id(self, seq_id):
        req_id = self.seq_id_to_req_id.get(seq_id, None)
        assert req_id is not None
        return req_id

    def _pad_to_max(self, x):
        z = torch.empty(x.shape[0], self.max_length, dtype=torch.int64)
        for idx, item in enumerate(x):
            pad = torch.zeros(self.max_length - len(x[idx]), dtype=torch.int)
            z[idx] = torch.cat((x[idx], pad))
        return z

    def _run_prefill(self, tokens, seq_ids):
        input_ids, attention_mask = tokens["input_ids"], tokens["attention_mask"]

        input_ids = self._pad_to_max(input_ids)
        attention_mask = self._pad_to_max(attention_mask)

        n_active_seqs, context_len = input_ids.shape
        cache_ids = (
            torch.arange(context_len)
            .reshape(1, context_len)
            .expand(n_active_seqs, context_len)
            .mul(attention_mask)
        )
        with torch.inference_mode():
            logits = self.model(
                input_ids, cache_ids=cache_ids, start_ids=torch.as_tensor(seq_ids)
            )
        next_tokens = select_tokens(logits)

        return next_tokens, cache_ids.max(dim=1, keepdim=True).values + 1

    def _run_decode(self, next_tokens, cache_ids, seq_ids):
        with torch.inference_mode():
            logits = self.model(next_tokens, cache_ids=cache_ids, start_ids=seq_ids)
        next_tokens = select_tokens(logits)

        return next_tokens

    def _clean_up(self, seq_id, req_id):
        # clean up
        del self.seq_id_to_req_id[seq_id]
        del self.context.cache[req_id]
        self.decode_seq_ids[seq_id] = -1
        self.decode_cache_ids[seq_id, :] = torch.zeros(1, dtype=torch.int64)
        self.decode_next_tokens[seq_id, :] = torch.tensor(
            [self.tokenizer.eos_token_id], dtype=torch.int64
        )
        # add seq_id back to self.empty_seq_ids
        self._add_empty_seq_id(seq_id)

    def _clean_dead_reqs(self):
        local_decode_seq_ids = torch.cat(torch.where(self.decode_seq_ids > -1))
        for _, seq_id in enumerate(local_decode_seq_ids):
            seq_id_value = seq_id.item()
            req_id = self._get_req_id(seq_id_value)
            if req_id not in self.context.request_ids:
                self._clean_up(seq_id_value, req_id)

    def _update_results(
        self,
        results,
        seq_id,
        idx,
        cache_ids,
        next_tokens,
        prefill_tokens=None,
        prefill_input_text=None,
    ):
        # 0: this seq_id is used for decoding if this slot is set 0
        self.decode_seq_ids[seq_id] = 0
        self.decode_cache_ids[seq_id, :] = cache_ids[idx, :]
        self.decode_next_tokens[seq_id, :] = next_tokens[idx, :]
        req_id = self._get_req_id(seq_id)
        cur_text = self.tokenizer.decode(next_tokens[idx, :], skip_special_tokens=False)
        if not (cur_text.startswith(" ") or cur_text.endswith(" ")):
            if prefill_tokens is None:
                previous_tokens = self.decode_next_tokens[seq_id, -1]
            else:
                previous_tokens = prefill_tokens["input_ids"][idx, -1]

            text = self.tokenizer.decode(
                torch.cat((torch.tensor([previous_tokens]), next_tokens[idx, :])),
                skip_special_tokens=False,
            )
            if text[: -len(cur_text)].endswith(" "):
                cur_text = " " + cur_text

        results[req_id] = {
            "text": cur_text
            if prefill_input_text is None
            else prefill_input_text[idx] + cur_text,
            "tokens": [next_tokens[idx, -1].item()],
        }

    def _create_stopping_criteria(self, req_id, seq_id, max_new_tokens):
        class StoppingCriteria(object):
            def __init__(
                self,
                outer,
                req_id,
                seq_id,
                stop_token,
                max_new_tokens,
            ):
                self.req_id = req_id
                self.seq_id = seq_id
                self.outer = outer
                self.max_new_tokens = max_new_tokens
                self.stop_token = stop_token

            def __call__(self, res):
                self.max_new_tokens -= 1

                if self.max_new_tokens == 0 or res["tokens"][-1] == self.stop_token:
                    self.outer._clean_up(self.seq_id, self.req_id)
                    return True
                return False

        return StoppingCriteria(
            outer=self,
            req_id=req_id,
            seq_id=seq_id,
            stop_token=self.tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
        )

    def _set_class(self, ctx):
        handler_config = ctx.model_yaml_config.get("handler", {})
        model_class_name = handler_config.get("model_class_name", None)

        assert (
            model_class_name
        ), "model_class_name not found in the section of handler in model config yaml file"
        model_module_prefix = handler_config.get("model_module_prefix", None)
        self.model_class = import_class(
            class_name=model_class_name,
            module_prefix=model_module_prefix,
        )

        tokenizer_class_name = handler_config.get("tokenizer_class_name", None)
        assert (
            tokenizer_class_name
        ), "tokenizer_class_name not found in the section of handler in model config yaml file"

        tokenizer_module_prefix = handler_config.get("tokenizer_module_prefix", None)

        self.tokenizer_class = import_class(
            class_name=tokenizer_class_name, module_prefix=tokenizer_module_prefix
        )
