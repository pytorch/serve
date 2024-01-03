import logging
import os

import torch
import torch_neuronx
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer
from transformers_neuronx.config import ContinuousBatchingConfig, NeuronConfig
from transformers_neuronx.llama.model import LlamaForSampling
from transformers_neuronx.module import save_pretrained_split
from transformers_neuronx.sampling import select_tokens

from ts.context import Context
from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)


class LlamaContinuousBatchingHandler(BaseHandler):
    def __init__(self):
        super(LlamaContinuousBatchingHandler, self).__init__()
        # the queue of seq_ids which are available for a new request
        self.batch_size = 2
        self.max_new_tokens = 25
        self.max_length = 100
        self.tokenizer = None
        self.decode_next_tokens = None
        self.decode_cache_ids = None
        self.decode_seq_ids = None
        self.empty_seq_ids = []
        # map seq_id to req_id
        self.seq_id_to_req_id = {}

    def initialize(self, ctx: Context):
        super().initialize(ctx)
        logger.info(f"Initialized {self.__class__}")

        model_dir = ctx.system_properties.get("model_dir")
        model_checkpoint_dir = ctx.model_yaml_config.get("handler", {}).get(
            "model_checkpoint_dir", ""
        )
        model_checkpoint_path = f"{model_dir}/{model_checkpoint_dir}"
        model_path = f'{model_dir}/{ctx.model_yaml_config["handler"]["model_path"]}'

        if not os.path.exists(model_checkpoint_path):
            # Load and save the CPU model
            model_cpu = AutoModelForCausalLM.from_pretrained(
                model_path, low_cpu_mem_usage=True
            )
            save_pretrained_split(model_cpu, model_checkpoint_path)
            # Load and save tokenizer for the model
            tokenizer = AutoTokenizer.from_pretrained(
                model_path, return_tensors="pt", padding_side="left"
            )
            tokenizer.save_pretrained(model_checkpoint_path)

        os.environ["NEURONX_CACHE"] = "on"
        os.environ["NEURON_COMPILE_CACHE_URL"] = f"{model_dir}/neuron_cache"
        os.environ["NEURON_CC_FLAGS"] = "-O1 --model-type=transformer"

        self.max_length = int(
            ctx.model_yaml_config.get("handler", {}).get("max_length", self.max_length)
        )
        self.max_new_tokens = int(
            ctx.model_yaml_config.get("handler", {}).get(
                "max_new_tokens", self.max_new_tokens
            )
        )
        self.batch_size = int(
            ctx.model_yaml_config.get("handler", {}).get("batch_size", self.batch_size)
        )

        # settings for model compilation and loading
        amp = ctx.model_yaml_config.get("handler", {}).get("amp", "fp32")
        tp_degree = ctx.model_yaml_config.get("handler", {}).get("tp_degree", 6)
        context_length_estimate = ctx.model_yaml_config.get("handler", {}).get(
            "context_length_estimate", self.max_length
        )

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

        self.tokenizer = LlamaTokenizer.from_pretrained(
            model_checkpoint_path, return_tensors="pt", padding_side="left"
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"

        continuous_batching_config = ContinuousBatchingConfig(
            batch_size_for_shared_caches=self.batch_size
        )
        neuron_config = NeuronConfig(continuous_batching=continuous_batching_config)
        self.model = LlamaForSampling.from_pretrained(
            model_checkpoint_path,
            batch_size=self.batch_size,
            amp=amp,
            tp_degree=tp_degree,
            n_positions=self.max_length,
            context_length_estimate=context_length_estimate,
            neuron_config=neuron_config,
        )
        logger.info("Starting to compile the model")
        self.model.to_neuron()
        logger.info("Model has been successfully compiled")

        # 1D: [seq_id]
        # an empty slot if seq_id is -1
        self.decode_seq_ids = torch.full([self.batch_size], -1)
        # 2D:[batch_size, next_cache_id]
        self.decode_cache_ids = torch.zeros(self.batch_size, 1, dtype=torch.int64)
        # 2D: [batch_size, next_token]
        self.decode_next_tokens = torch.zeros(self.batch_size, 1, dtype=torch.int64)

        for seq_id, batch_id in enumerate(reversed(range(self.batch_size))):
            self.empty_seq_ids.append(batch_id)

        logger.info("Model %s loaded successfully", ctx.model_name)
        self.initialized = True

    def preprocess(self, requests):
        prefill_req_ids, prefill_seq_ids, prefill_input_text, decode_seq_ids = (
            [],
            [],
            [],
            [],
        )
        for req_id, req_data in zip(self.context.request_ids.values(), requests):
            if not req_id in self.context.cache:
                prefill_req_ids.append(req_id)
                seq_id = self._get_empty_seq_id()
                prefill_seq_ids.append(seq_id)

                data = req_data["data"]
                if isinstance(data, (bytes, bytearray)):
                    data = data.decode("utf-8")
                    max_new_tokens = int(
                        req_data.get("max_new_tokens", self.max_new_tokens)
                    )
                    prefill_input_text.append(data.strip())

                    self.context.cache[req_id] = {
                        "seq_id": seq_id,
                        "stopping_criteria": self._create_stopping_criteria(
                            req_id=req_id, seq_id=seq_id, max_new_tokens=max_new_tokens
                        ),
                    }
            else:
                decode_seq_ids.append(self.context.cache[req_id]["seq_id"])

        prefill_tokens = None
        if len(prefill_req_ids) > 0:
            prefill_tokens = self.tokenizer(
                prefill_input_text, return_tensors="pt", padding=True
            )
        return prefill_tokens, prefill_seq_ids, decode_seq_ids

    def inference(self, inputs):
        prefill_tokens, prefill_seq_ids, req_decode_seq_ids = inputs
        results = {}
        # Test if this is the beginning of a continuous batching
        go_to_decode = True if len(req_decode_seq_ids) > 0 else False
        if len(prefill_seq_ids) > 0:
            prefill_next_tokens, prefill_cache_ids = self._run_prefill(
                prefill_tokens, prefill_seq_ids
            )
            for i, prefill_seq_id in enumerate(prefill_seq_ids):
                self._update_results(
                    results, prefill_seq_id, i, prefill_cache_ids, prefill_next_tokens
                )

        if go_to_decode:
            decode_seq_ids = torch.where(self.decode_seq_ids > -1)
            decode_cache_ids = torch.where(self.decode_cache_ids > 0)
            decode_next_tokens = torch.where(self.decode_next_tokens > 0)
            next_tokens = self._run_decode(
                decode_next_tokens, decode_cache_ids, decode_seq_ids
            )

            filter_prefill_seq_ids = (
                torch.isin(decode_seq_ids, torch.as_tensor(prefill_seq_ids))
                if len(prefill_seq_ids) > 0
                else torch.full(decode_seq_ids.shape, False)
            )

            decode_cache_ids = decode_cache_ids + 1
            for i, is_prefill_seq_id in enumerate(filter_prefill_seq_ids):
                if not is_prefill_seq_id:
                    seq_id = decode_seq_ids[i]
                    if seq_id in req_decode_seq_ids:
                        self._update_results(
                            results, seq_id, i, decode_cache_ids, next_tokens
                        )
                    else:
                        req_id = self._get_req_id(seq_id)
                        logger.warning(
                            f"Found request id:{req_id} in cache, but not in batch requests. Delete it"
                        )
                        self.clean_up(self, seq_id, req_id)

        return results

    def postprocess(self, inference_output):
        self.context.stopping_criteria = [
            self.batch_store[self._get_seq_id(req_id)]["stopping_criteria"](
                req_id,
            )
            for req_id in self.context.request_ids.values()
        ]

        return inference_output

    def _get_empty_seq_id(self):
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
        req_id = self.seq_id_to_req_id(seq_id, None)
        assert req_id is not None
        return req_id

    def _pad_to_max(self, x):
        for idx, item in enumerate(x):
            x[idx] = x[idx] + [0] * (self.max_length - len(x[idx]))
        return x

    def _run_prefill(self, tokens, seq_ids):
        input_ids, attention_mask = tokens["input_ids"], tokens["attention_mask"]
        logger.info(
            f"before padding: input_ids={input_ids}, attention_mask={attention_mask}"
        )
        input_ids = self._pad_to_max(input_ids)
        attention_mask = self._pad_to_max(attention_mask)
        logger.info(
            f"after padding: input_ids={input_ids}, attention_mask={attention_mask}"
        )
        n_active_seqs, context_len = input_ids.shape
        cache_ids = (
            torch.arange(context_len)
            .reshape(1, context_len)
            .expand(n_active_seqs, context_len)
            .mul(attention_mask)
        )
        with torch.inference_mode():
            logits = self.model(
                input_ids, cache_ids=self.cache_ids, start_ids=torch.as_tensor(seq_ids)
            )
        next_tokens = select_tokens(logits)

        return next_tokens, cache_ids.max(dim=1, keepdim=True).values + 1

    def _run_decode(self, next_tokens, cache_ids, seq_ids):
        with torch.inference_mode():
            logits = self.model(next_tokens, cache_ids=cache_ids, start_ids=seq_ids)
        next_tokens = select_tokens(logits)
        return next_tokens

    def clean_up(self, seq_id, req_id):
        # clean up
        del self.seq_id_to_req_id[seq_id]
        del self.context.cache[req_id]
        self.decode_seq_ids[seq_id] = -1
        self.decode_cache_ids[seq_id, :] = torch.zeros(
            1, dtype=torch.int64, device="cpu"
        )
        self.decode_next_tokens[seq_id, :] = torch.zeros(
            1, dtype=torch.int64, device="cpu"
        )

        # add seq_id back to self.empty_seq_ids
        self._add_empty_seq_id(seq_id)

    def _update_results(self, results, seq_id, idx, cache_ids, next_tokens):
        self.decode_cache_ids[seq_id, :] = cache_ids[idx, :]
        self.decode_next_tokens[seq_id, :] = next_tokens[idx, :]
        req_id = self._get_req_id(seq_id)
        self.seq_id_to_req_id[seq_id] = req_id
        results[req_id] = {
            "text": self.tokenizer.decode(
                next_tokens[idx, -1], skip_special_tokens=True
            ),
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

                if self.max_new_tokens == 0 or res["ids"][-1] == self.stop_token:
                    self.outer.clean_up(self.req_id, self.seq_id)
                    return True
                return False

        return StoppingCriteria(
            outer=self,
            req_id=req_id,
            seq_id=seq_id,
            stop_token=self.tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
        )
