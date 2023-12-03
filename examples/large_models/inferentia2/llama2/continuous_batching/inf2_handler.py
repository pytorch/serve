import logging
import os
import types
from abc import ABC

import torch
import torch_neuronx
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer
from transformers_neuronx.generation_utils import HuggingFaceGenerationModelAdapter
from transformers_neuronx.llama.model import LlamaForSampling
from transformers_neuronx.module import save_pretrained_split

from ts.context import Context
from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)


class LlamaHandler(BaseHandler, ABC):
    """
    Transformers handler class for sequence, token classification and question answering.
    """

    def __init__(self):
        super(LlamaHandler, self).__init__()
        self.max_length = None
        self.max_new_tokens = None
        self.tokenizer = None
        self.batch_size = 1
        self.initialized = False
        self.batch_ids = []

    def initialize(self, ctx: Context):
        """In this initialize function, the HF large model is loaded and
        partitioned using DeepSpeed.
        Args:
            ctx (context): It is a JSON Object containing information
            pertaining to the model artifacts parameters.
        """
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
            tokenizer = AutoTokenizer.from_pretrained(model_path, return_tensors="pt")
            tokenizer.save_pretrained(model_checkpoint_path)

        os.environ["NEURONX_CACHE"] = "on"
        os.environ["NEURON_COMPILE_CACHE_URL"] = f"{model_dir}/neuron_cache"
        os.environ["NEURON_CC_FLAGS"] = "-O1 --model-type=transformer"

        # settings for model compiliation and loading
        amp = ctx.model_yaml_config.get("handler", {}).get("amp", "fp32")
        tp_degree = ctx.model_yaml_config.get("handler", {}).get("tp_degree", 6)
        context_length_estimate = ctx.model_yaml_config.get("handler", {}).get(
            "context_length_estimate", None
        )

        self.max_length = int(ctx.model_yaml_config["handler"]["max_length"])
        self.max_new_tokens = ctx.model_yaml_config["handler"]["max_new_tokens"]
        self.batch_size = int(
            ctx.model_yaml_config.get("handler", {}).get("batch_size", 1)
        )

        for i in range(self.batch_size):
            self.batch_ids.append(i)

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

        self.tokenizer = LlamaTokenizer.from_pretrained(model_checkpoint_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"
        self.model = LlamaForSampling.from_pretrained(
            model_checkpoint_path,
            batch_size=self.batch_size,
            amp=amp,
            tp_degree=tp_degree,
            n_positions=self.max_length if self.max_length > 2048 else 2048,
            context_length_estimate=context_length_estimate,
        )
        logger.info("Starting to compile the model")
        self.model.to_neuron()
        logger.info("Model has been successfully compiled")
        model_config = AutoConfig.from_pretrained(model_checkpoint_path)
        self.model = HuggingFaceGenerationModelAdapter(model_config, self.model)

        # Replace _update_model_kwargs_for_generation of model with a method that extracts the kv cache for us
        old_update = self.model._update_model_kwargs_for_generation
        ctx.cache = {}
        ctx.kv_cache = {}

        def extract_past_key_values_func(self, *args, **kwargs):
            ctx.kv_cache["past_key_values"] = args[0]["past_key_values"]
            return old_update(*args, **kwargs)

        self.model._update_model_kwargs_for_generation = types.MethodType(
            extract_past_key_values_func, self.model
        )

        logger.info("Model %s loaded successfully", ctx.model_name)
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
        logger.info(f"requests size={len(requests)}")

        prefill_req_ids, decode_req_ids, prefill_input = [], [], []
        for req_id, req_data in zip(self.context.request_ids.values(), requests):
            # Tokenizer requests which are not prefilled yet
            if not req_id in self.context.cache:
                data = req_data["data"]
                if isinstance(data, (bytes, bytearray)):
                    data = data.decode("utf-8")
                    max_new_tokens = int(
                        req_data.get("max_new_tokens", str(self.max_new_tokens))
                    )
                    prefill_input.append(
                        {"data": data.strip(), "max_new_token": max_new_tokens}
                    )
                logger.info("Received text: '%s'", data)
                prefill_req_ids.append(req_id)
            else:
                decode_req_ids.append(req_id)

        prefill_encoded = None
        if len(prefill_req_ids) > 0:
            prefill_encoded = self._run_tokenizer_batch(
                prefill_input, prefill_req_ids, decode_req_ids
            )

        return prefill_req_ids, prefill_encoded, decode_req_ids

    def inference(self, input_batch):
        """
        Predicts the class (or classes) of the received text using the serialized transformers
        checkpoint.
        Args:
            input_batch (tuple): A tuple with two tensors: the batch of input ids and the batch
                                of attention masks, as returned by the preprocess function.
        Returns:
            list: A list of strings with the predicted values for each input text in the batch.
        """

        prefill_req_ids, prefill_encoded, decode_req_ids = input_batch

        results = {}
        if prefill_req_ids:
            # Prefill requests
            results.update(self._run_prefill_batch(prefill_req_ids, prefill_encoded))
        else:
            # Decode the rest
            results.update(self._run_decode(decode_req_ids))

        return [results[i] for i in self.context.request_ids.values()]

    def postprocess(self, inference_output):
        """Post Process Function converts the predicted response into Torchserve readable format.
        Args:
            inference_output (list): It contains the predicted response of the input text.
        Returns:
            (list): Returns a list of the Predictions and Explanations.
        """
        self.context.stopping_criteria = [
            self.context.cache[i]["stopping_criteria"]
            for i in self.context.request_ids.values()
        ]

        return inference_output

    def _run_tokenizer_batch(self, prefill_input, prefill_req_ids, decode_req_ids):
        # Pad input to match compiled model batch size
        prefill_input_text = [""] * self.batch_size
        for req_id, input_data in zip(prefill_req_ids, prefill_input):
            idx = self.batch_ids.pop()
            prefill_input_text[idx] = input_data["data"]
            self.context.cache[req_id] = {"batch_idx": idx}

        batch_encoded = self.tokenizer(
            prefill_input_text,
            return_tensors="pt",
            padding=True,
            add_special_tokens=True,
            return_token_type_ids=False,
            truncation=True,
        )
        for idx, req_id in enumerate(prefill_req_ids):
            encoded = {
                "input_ids": batch_encoded["input_ids"][idx],
                "attention_mask": batch_encoded["attention_mask"][idx],
            }
            self.context.cache[req_id].update(
                {
                    "stopping_criteria": self._create_stopping_criteria(
                        req_id, max_new_tokens=prefill_input[idx]["max_new_tokens"]
                    ),
                    "encoded": encoded,
                    "prompt_length": encoded["input_ids"].shape[0],
                }
            )

        for req_id in decode_req_ids:
            idx = self.context.cache[req_id]["batch_idx"]
            batch_encoded["input_ids"][idx] = self.context.cache[req_id]["encoded"][
                "input_ids"
            ]
            batch_encoded["attention_mask"][idx] = self.context.cache[req_id][
                "encoded"
            ]["attention_mask"]

        return batch_encoded

    @torch.no_grad()
    def _run_prefill_batch(self, prefill_req_ids, prefill_encoded):
        self.model.reset_generation()
        outputs = self.model.generate(
            prefill_encoded["input_ids"],
            prefill_encoded["attention_mask"],
            max_new_tokens=1,
            return_dict_in_generate=True,
            use_cache=True,
        )

        outputs_decoded = self.tokenizer.batch_decode(
            outputs.sequences, skip_special_tokens=True
        )

        # Prefill requests
        results = {}
        for idx, req_id in enumerate(prefill_req_ids):
            self.context.cache[req_id]["encoded"]["input_ids"] = outputs.sequences[idx]
            device = next(iter(self.model.parameters())).device
            dtype = torch.int64
            config = {"device": device, "dtype": dtype}
            attention_mask = self.context.cache[req_id]["encoded"]["attention_mask"]
            attention_mask = torch.concat(
                (attention_mask, torch.ones((1), **config)), dim=0
            )
            self.context.cache[req_id]["encoded"]["attention_mask"] = attention_mask

            results[req_id] = {
                "text": outputs_decoded[idx],
                "tokens": outputs.sequences[idx].tolist(),
            }
        return results

    def _run_decode(self, ids):
        encoded = self._prepare_model_inputs(ids)

        outputs = self.model.generate(
            **encoded, max_new_tokens=1, return_dict_in_generate=True, use_cache=True
        )

        device = next(iter(self.model.parameters())).device
        dtype = torch.int64
        config = {"device": device, "dtype": dtype}

        results = {}
        for idx, req_id in enumerate(ids):
            if req_id == "batch_padding":
                continue

            self.context.cache[req_id]["encoded"]["input_ids"] = outputs.sequences[idx]
            attention_mask = encoded["attention_mask"][idx]
            attention_mask = torch.concat(
                (attention_mask, torch.ones((1), **config)), dim=0
            )
            self.context.cache[req_id]["encoded"]["attention_mask"] = attention_mask
            results[req_id] = {
                "text": self.tokenizer.decode(
                    outputs.sequences[idx][-1], skip_special_tokens=True
                ),
                "tokens": [outputs.sequences[idx][-1].item()],
            }
        return results

    def _prepare_model_inputs(self, ids):
        lengths = list(
            torch.sum(self.context.cache[i]["encoded"]["attention_mask"]).item()
            for i in ids
        )
        max_len = max(lengths)

        for idx in range(self.batch_size - len(ids)):
            ids.append("batch_padding")
            lengths.append(0)

        device = next(iter(self.model.parameters())).device
        dtype = torch.int64
        config = {"device": device, "dtype": dtype}

        input_ids = []
        attention_mask = []

        for req_id, seq_len in zip(ids, lengths):
            if req_id != "batch_padding":
                input_ids.append(self.context.cache[req_id]["encoded"]["input_ids"])
                attention_mask.append(
                    self.context.cache[req_id]["encoded"]["attention_mask"]
                )

            else:
                config = {"device": device, "dtype": dtype}
                input_ids.append(
                    self.tokenizer.pad_token_id + torch.zeros((max_len), **config)
                )
                attention_mask.append(torch.zeros((max_len), **config))

            padded_len = input_ids[-1].size()[-1]
            logger.info(f"req_id={req_id}, padded_len={padded_len}, max_len={max_len}")
            if padded_len < max_len:
                # Apply padding to input_ids, attention_mask and past_key_values
                n = max_len - seq_len
                input_ids[-1] = torch.concat(
                    (
                        self.tokenizer.pad_token_id + torch.zeros((n), **config),
                        input_ids[-1],
                    )
                )
                attention_mask[-1] = torch.concat(
                    (torch.zeros((n), **config), attention_mask[-1])
                )
            elif padded_len > max_len:
                # Truncate padding from input_ids, attention_mask and past_key_values
                logger.info(f"padded_len shape={input_ids[-1].size()}")
                input_ids[-1] = input_ids[-1][-max_len:]
                attention_mask[-1] = attention_mask[-1][-max_len:]

        encoded = {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_mask),
        }
        return encoded

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

                if self.max_new_tokens == 0 or res["tokens"][-1] == self.stop_token:
                    self.clean_up()
                    return True
                return False

            def clean_up(self):
                del self.cache[self.req_id]

        return StoppingCriteria(
            self.context.cache,
            req_id,
            self.tokenizer.eos_token_id,
            max_new_tokens,
        )
