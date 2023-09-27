import logging
import types
from abc import ABC

import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from ts.context import Context
from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)
logger.info("Transformers version %s", transformers.__version__)


class LlamaHandler(BaseHandler, ABC):
    """
    Transformers handler class for sequence, token classification and question answering.
    """

    def __init__(self):
        super(LlamaHandler, self).__init__()
        self.max_length = None
        self.max_new_tokens = None
        self.tokenizer = None
        self.initialized = False

    def initialize(self, ctx: Context):
        """In this initialize function, the HF large model is loaded and
        partitioned using DeepSpeed.
        Args:
            ctx (context): It is a JSON Object containing information
            pertaining to the model artifacts parameters.
        """
        model_dir = ctx.system_properties.get("model_dir")
        self.max_length = int(ctx.model_yaml_config["handler"]["max_length"])
        self.max_new_tokens = int(ctx.model_yaml_config["handler"]["max_new_tokens"])
        model_name = ctx.model_yaml_config["handler"]["model_name"]
        model_path = f'{model_dir}/{ctx.model_yaml_config["handler"]["model_path"]}'
        seed = int(ctx.model_yaml_config["handler"]["manual_seed"])
        torch.manual_seed(seed)

        logger.info("Model %s loading tokenizer", ctx.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="balanced",
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            load_in_8bit=True,
            trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.add_special_tokens(
            {
                "pad_token": "<PAD>",
            }
        )
        self.model.resize_token_embeddings(self.model.config.vocab_size + 1)

        # Replace _update_model_kwargs_for_generation of model with a method that extracts the kv cache for us
        old_update = self.model._update_model_kwargs_for_generation
        ctx.cache = {}
        ctx.kv_cache = {}

        def extract_past_key_values_func(self, *args, **kwargs):
            ctx.kv_cache["past_key_values"] = args[0]["past_key_values"]
            return old_update(*args, **kwargs)

        self.model._update_model_kwargs_for_generation = types.MethodType(extract_past_key_values_func, self.model)

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

        # why 2?
        assert len(self.context.request_ids.values()) <= 2
        self._clean_cache()

        prefill, decode = [], []
        for req_id, req_data in zip(self.context.request_ids.values(), requests):
            # Tokenizer requests which are not prefilled yet
            if not req_id in self.context.cache:
                data = req_data["body"] or req_data["data"]
                encoded = self.encode_input_text(data)
                encoded["past_key_values"] = None
                self.context.cache[req_id] = {
                    #"stopping_criteria": self._create_stopping_criteria(req_id, max_new_tokens=data["max_new_tokens"]),
                    "stopping_criteria": self._create_stopping_criteria(req_id),
                    "encoded": encoded,
                    "prompt_length": len(encoded["input_ids"]),
                }
                prefill.append(req_id)
            else:
                decode.append(req_id)
        return prefill, decode

    def encode_input_text(self, input_text):
        """
        Encodes a single input text using the tokenizer.
        Args:
            input_text (str): The input text to be encoded.
        Returns:
            tuple: A tuple with two tensors: the encoded input ids and the attention mask.
        """
        if isinstance(input_text, (bytes, bytearray)):
            input_text = input_text.decode("utf-8")
        logger.info("Received text: '%s'", input_text)
        inputs = self.tokenizer.encode_plus(
            input_text,
            max_length=self.max_length,
            padding=True,
            add_special_tokens=True,
            return_tensors="pt",
            truncation=True,
        )

        return inputs

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

        prefill, decode_ids = input_batch

        # Prefill requests
        results = {}
        for req_id in prefill:
            results[req_id] = self._run_prefill(req_id)

        # Decode the rest
        decode_result = self._run_decode(decode_ids) if decode_ids else {}
        results.update(decode_result)
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

    def _run_prefill(self, req_id):
        assert self.context.cache[req_id]["encoded"]["past_key_values"] is None, "There should be no cached values"
        output = self.model.generate(
            **self.context.cache[req_id]["encoded"],
            max_new_tokens=1,
            return_dict_in_generate=True,
            use_cache=True
        )
        # Save extracted kv cache values and adjust attention mask for next call
        self.context.cache[req_id]["encoded"]["past_key_values"] = self.context.kv_cache["past_key_values"]
        del self.context.kv_cache["past_key_values"]
        self.context.cache[req_id]["encoded"]["input_ids"] = output.sequences
        attention_mask = self.context.cache[req_id]["encoded"]["attention_mask"]
        attention_mask = torch.concat((attention_mask, torch.ones((1, 1), dtype=torch.int64)), dim=1)
        self.context.cache[req_id]["encoded"]["attention_mask"] = attention_mask

        result = {
            "text": self.tokenizer.decode(output.sequences[0], skip_special_tokens=True),
            "ids": output.sequences[0].tolist(),
        }
        return result

    def _run_decode(self, ids):
        assert len(ids)

        encoded = self._prepare_model_inputs(ids)

        outputs = self.model.generate(
            **encoded,
            max_new_tokens=1,
            return_dict_in_generate=True,
            use_cache=True
        )

        results = {}
        for idx, req_id in enumerate(ids):
            self.context.cache[req_id]["encoded"]["past_key_values"] = self._collect_kv_cache_of_idx_in_batch(idx)
            self.context.cache[req_id]["encoded"]["input_ids"] = outputs.sequences[idx].unsqueeze(0)
            attention_mask = encoded["attention_mask"][idx].unsqueeze(0)
            attention_mask = torch.concat((attention_mask, torch.ones((1, 1), dtype=torch.int64)), dim=1)
            self.context.cache[req_id]["encoded"]["attention_mask"] = attention_mask
            results[req_id] = {
                "text": self.tokenizer.decode(outputs.sequences[idx][-1], skip_special_tokens=True),
                "ids": [outputs.sequences[idx][-1].item()],
            }
        del self.context.kv_cache["past_key_values"]
        return results

    def _prepare_model_inputs(self, ids):
        lengths = list(torch.sum(self.context.cache[i]["encoded"]["attention_mask"], dim=1).item() for i in ids)
        max_len = max(lengths)

        input_ids = []
        attention_mask = []
        kv_cache = {}
        for req_id, seq_len in zip(ids, lengths):
            input_ids.append(self.context.cache[req_id]["encoded"]["input_ids"])
            attention_mask.append(self.context.cache[req_id]["encoded"]["attention_mask"])

            for layer_idx, layer_kv in enumerate(self.context.cache[req_id]["encoded"]["past_key_values"]):
                k, v = layer_kv
                kv_cache[layer_idx] = kv_cache.get(layer_idx, {})
                kv_cache[layer_idx][0] = kv_cache.get(layer_idx, {}).get(0, []) + [k]
                kv_cache[layer_idx][1] = kv_cache.get(layer_idx, {}).get(1, []) + [v]
            padded_len = input_ids[-1].size()[-1]
            if padded_len < max_len:
                # Apply padding to input_ids, attention_mask and past_key_values
                n = max_len - seq_len
                input_ids[-1] = torch.concat(
                    (self.tokenizer.pad_token_id + torch.zeros((1, n), dtype=torch.int64), input_ids[-1]), dim=1)
                attention_mask[-1] = torch.concat((torch.zeros((1, n), dtype=torch.int64), attention_mask[-1]), dim=1)

                size_delta = list(kv_cache[0][0][-1].size())
                size_delta[2] = n
                dtype = kv_cache[0][0][-1].dtype
                for layer_idx in range(len(kv_cache)):
                    kv_cache[layer_idx][0][-1] = torch.concat(
                        (torch.zeros(size_delta, dtype=dtype), kv_cache[layer_idx][0][-1]), dim=2)
                    kv_cache[layer_idx][1][-1] = torch.concat(
                        (torch.zeros(size_delta, dtype=dtype), kv_cache[layer_idx][1][-1]), dim=2)

            elif padded_len > max_len:
                # Truncate padding from input_ids, attention_mask and past_key_values
                input_ids[-1] = input_ids[-1][:, -max_len:]
                attention_mask[-1] = attention_mask[-1][:, -max_len:]

                for layer_idx in range(len(kv_cache)):
                    kv_cache[layer_idx][0][-1] = kv_cache[layer_idx][0][-1][:, :, (-max_len + 1):, :]
                    kv_cache[layer_idx][1][-1] = kv_cache[layer_idx][1][-1][:, :, (-max_len + 1):, :]
            del self.context.cache[req_id]["encoded"]["past_key_values"]

        for layer_idx in range(len(kv_cache)):
            kv_cache[layer_idx][0] = torch.concat(kv_cache[layer_idx][0], dim=0)
            kv_cache[layer_idx][1] = torch.concat(kv_cache[layer_idx][1], dim=0)

        kv_cache = tuple((kv_cache[layer_idx][0], kv_cache[layer_idx][1]) for layer_idx in range(len(kv_cache)))

        encoded = {
            "input_ids": torch.concat(input_ids, dim=0),
            "attention_mask": torch.concat(attention_mask, dim=0),
            "past_key_values": kv_cache,
        }
        return encoded

    def _collect_kv_cache_of_idx_in_batch(self, idx):
        # The materialization of the tuple here is important for some reason (TODO: figure out why); Otherwise prediction differ
        return tuple(
            tuple(kv[idx, ...].unsqueeze(0) for kv in layers) for layers in self.context.kv_cache["past_key_values"])

    def _create_stopping_criteria(self, req_id, max_new_tokens=25):
        class StoppingCriteria(object):
            def __init__(
                    self, cache, req_id, stop_token, max_new_tokens,
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
            self.tokenizer.eos_token_id,
            max_new_tokens,
        )

    def _clean_cache(self):
        new_ids = set(self.context.request_ids.keys())
        for idx in self.context.kv_cache.keys():
            if idx not in new_ids:
                del self.context.kv_cache[idx]
