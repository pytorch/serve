import logging
import os
import re

import torch
import transformers
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    T5ForConditionalGeneration,
)

from ts.context import Context
from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)
logger.info("PyTorch version %s", torch.__version__)
logger.info("Transformers version %s", transformers.__version__)

IPEX_ENABLE = False
if os.environ.get("TS_IPEX_ENABLE", "false") == "true":
    try:
        import intel_extension_for_pytorch as ipex

        try:
            ipex._C.disable_jit_linear_repack()
            torch._C._jit_set_texpr_fuser_enabled(False)
        except Exception:
            pass
        IPEX_ENABLE = True
        logger.info("IPEX optimization is enabled")
        logger.info("IPEX version %s", ipex.__version__)

    except ImportError as error:
        logger.warning(
            "IPEX is enabled but intel-extension-for-pytorch cannot be imported. Proceeding without IPEX"
        )
        IPEX_ENABLE = False
else:
    logger.warning(
        "IPEX is not enabled, consider enabling it for best performance on Intel hardware"
    )


class IpexLLMHandler(BaseHandler):
    def __init__(self):
        super(IpexLLMHandler, self).__init__()

        # for streaming the generated texts back to client
        self.output_streamer = None

    def initialize(self, ctx: Context):
        model_name = ctx.model_yaml_config["handler"]["model_name"]
        # path to quantized model, if we are quantizing on the fly, we'll use this path to save the model
        self.clear_cache_dir = ctx.model_yaml_config["handler"].get(
            "clear_cache_dir", False
        )
        self.quantized_model_path = ctx.model_yaml_config["handler"].get(
            "quantized_model_path", "best_model.pt"
        )
        self.example_inputs_mode = ctx.model_yaml_config["handler"].get(
            "example_inputs_mode", "MASK_KV_POS"
        )
        self.to_channels_last = ctx.model_yaml_config["handler"].get(
            "to_channels_last", False
        )

        # generation params
        self.batch_size = int(ctx.model_yaml_config["handler"].get("batch_size", "1"))
        self.input_tokens = int(
            ctx.model_yaml_config["handler"].get("input_tokens", "1024")
        )
        self.max_new_tokens = int(
            ctx.model_yaml_config["handler"].get("max_new_tokens", "128")
        )

        # enable auto mix precision
        self.auto_mixed_precision = ctx.model_yaml_config["handler"].get(
            "auto_mixed_precision", True
        )

        # use int8 bf16 mix
        self.quant_with_amp = ctx.model_yaml_config["handler"].get(
            "quant_with_amp", True
        )

        # WoQ related optimization params
        self.ipex_weight_only_quantization = ctx.model_yaml_config["handler"].get(
            "ipex_weight_only_quantization", False
        )
        if self.ipex_weight_only_quantization:
            self.woq_dtype = ctx.model_yaml_config["handler"].get("woq_dtype", "INT8")
            self.lowp_mode = ctx.model_yaml_config["handler"].get("lowp_mode", "BF16")
            self.act_quant_mode = ctx.model_yaml_config["handler"].get(
                "act_quant_mode", "PER_IC_BLOCK"
            )  # This is only relevant for INT4x2 quantization
            self.group_size = int(
                ctx.model_yaml_config["handler"].get("group_size", "-1")
            )

        # SQ related optimization params
        self.ipex_smooth_quantization = ctx.model_yaml_config["handler"].get(
            "ipex_smooth_quantization", False
        )
        if self.ipex_smooth_quantization:
            self.num_calib_iters = int(
                ctx.model_yaml_config["handler"].get("num_calibration_iters", 32)
            )
            self.alpha = float(ctx.model_yaml_config["handler"].get("alpha", 0.9))

        # Keeping outside because we want to use it for tracing as well
        self.calib_dataset = ctx.model_yaml_config["handler"].get(
            "calibration_dataset", "NeelNanda/pile-10k"
        )
        self.calib_split = ctx.model_yaml_config["handler"].get(
            "calibration_split", "train"
        )

        # decoding parameters
        self.greedy = ctx.model_yaml_config["handler"].get("greedy", False)

        # amp datatype
        if self.quant_with_amp or self.auto_mixed_precision:
            self.amp_enabled = True
            self.amp_dtype = torch.bfloat16
        else:
            self.amp_enabled = False
            self.amp_dtype = torch.float32

        # generate args: using greedy for now
        self.num_beams = 1 if self.greedy else 4
        # donot use min number of tokens on demo mode, only use it on benchmark mode
        self.generate_kwargs = dict(
            do_sample=False,
            temperature=0.9,
            num_beams=self.num_beams,
            max_new_tokens=self.max_new_tokens,
        )

        # device
        device = torch.device("cpu")

        # model config
        config = AutoConfig.from_pretrained(
            model_name, torchscript=True, trust_remote_code=True
        )

        # set up max context
        if not hasattr(config, "text_max_length"):
            config.text_max_length = int(self.input_tokens) + int(self.max_new_tokens)
        if "mpt" in model_name and not hasattr(config, "max_seq_len"):
            config.max_seq_len = int(self.input_tokens) + int(self.max_new_tokens)

        # load model and tokenizer,
        # We need special provision for t5 because it's seq2seq model, and can not be loaded with AutoModelForCausalLM
        if re.search("t5", config.architectures[0], re.IGNORECASE):
            self.user_model = T5ForConditionalGeneration.from_pretrained(
                model_name,
                config=config,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float,
            )
            input_ids = torch.ones(32).to(torch.long).unsqueeze(0)
            attention_mask = torch.ones_like(input_ids)
            dummy_inputs = self.user_model.prepare_inputs_for_generation(
                input_ids, attention_mask=attention_mask
            )
            if dummy_inputs.get("position_ids", None) is not None:
                self.example_inputs_mode = "MASK_KV_POS"

            # we also need to update generation kwargs
            self.generate_kwargs["max_length"] = self.generate_kwargs["max_new_tokens"]
            self.generate_kwargs.pop("max_new_tokens")

        else:
            self.user_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                config=config,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float,
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )

        logger.info("Data type of the model: %s", self.user_model.dtype)

        if self.to_channels_last:
            self.user_model = self.user_model.to(memory_format=torch.channels_last)
        self.user_model.eval()

        # dummy past key value
        self.beam_idx_tmp = torch.zeros(
            (2048, int(self.batch_size * self.num_beams)), dtype=torch.long
        ).contiguous()

        def _get_target_nums(names):
            for n in names:
                if hasattr(self.user_model.config, n):
                    return getattr(self.user_model.config, n)
            logger.error(f"Not found target {names[0]}")
            exit(1)

        num_heads_names = ["num_attention_heads", "n_head", "num_heads", "n_heads"]
        num_layers_names = ["num_hidden_layers", "n_layer", "num_layers", "n_layers"]
        hidden_size_names = ["hidden_size", "n_embd", "d_model"]
        n_heads = _get_target_nums(num_heads_names)
        n_layers = _get_target_nums(num_layers_names)
        hidden_size = _get_target_nums(hidden_size_names)
        head_dim = int(hidden_size / n_heads)
        self.global_past_key_value = [
            (
                torch.zeros(1, 0, 0, 1, dtype=torch.long).contiguous(),
                torch.zeros([1, n_heads, 1, head_dim]).contiguous(),
                torch.zeros([1, n_heads, 1, head_dim]).contiguous(),
                self.beam_idx_tmp,
            )
            for i in range(n_layers)
        ]

        logger.info(
            f"num_attention_heads: {n_heads}, num_hidden_layers: {n_layers}, hidden size: {hidden_size}, head_dim: {head_dim}"
        )

        logger.info("Preparing the dataset for calibration and tracing")

        class Evaluator:
            def __init__(
                self,
                user_model,
                example_inputs_mode,
                global_past_key_value,
                dataset,
                tokenizer,
                batch_size=1,
                num_beams=1,
                pad_val=1,
                pad_max=512,
            ):
                self.user_model = user_model
                self.example_inputs_mode = example_inputs_mode
                self.global_past_key_value = global_past_key_value
                self.dataset = dataset
                self.tokenizer = tokenizer
                self.batch_size = batch_size
                self.num_beams = num_beams

                self.pad_val = pad_val
                self.pad_max = pad_max
                self.dataset = self.dataset.map(self.tokenize_function, batched=True)
                self.dataset.set_format(type="torch", columns=["input_ids"])

            @torch.no_grad()
            def tokenize_function(self, examples):
                if "text" in examples:
                    example = self.tokenizer(examples["text"])
                elif "prompt" in examples:
                    example = self.tokenizer(examples["prompt"])
                elif "code" in examples:
                    example = self.tokenizer(examples["code"])
                return example

            @torch.no_grad()
            def collate_batch(self, batch):
                position_ids_padded = []
                input_ids_padded = []
                last_ind = []
                attention_mask_padded = []

                for text in batch:
                    input_ids = text["input_ids"]
                    last_ind.append(input_ids.shape[0] - 1)
                    attention_mask = torch.ones(len(input_ids))
                    position_ids = torch.arange(len(input_ids))

                    input_ids_padded.append(input_ids)
                    attention_mask_padded.append(attention_mask)
                    position_ids_padded.append(position_ids)

                if self.example_inputs_mode == "MASK_POS_KV":
                    model_inputs = (
                        torch.vstack(input_ids_padded),
                        torch.vstack(attention_mask_padded),
                        torch.vstack(position_ids_padded),
                        tuple(self.global_past_key_value),
                    )
                elif self.example_inputs_mode == "MASK_KV_POS":
                    model_inputs = (
                        torch.vstack(input_ids_padded),
                        torch.vstack(attention_mask_padded),
                        tuple(self.global_past_key_value),
                        torch.vstack(position_ids_padded),
                    )
                elif self.example_inputs_mode == "KV_MASK":
                    model_inputs = (
                        torch.vstack(input_ids_padded),
                        tuple(self.global_past_key_value),
                        torch.vstack(attention_mask_padded),
                    )
                elif self.example_inputs_mode == "MASK_KV":
                    model_inputs = (
                        torch.vstack(input_ids_padded),
                        torch.vstack(attention_mask_padded),
                        tuple(self.global_past_key_value),
                    )
                elif self.example_inputs_mode == "MASK_KV_ENC":
                    model_kwargs = {
                        "attention_mask": torch.vstack(attention_mask_padded),
                    }
                    model_kwargs = (
                        self.user_model._prepare_encoder_decoder_kwargs_for_generation(
                            torch.vstack(input_ids_padded), model_kwargs, "input_ids"
                        )
                    )
                    (
                        input_ids,
                        example_inputs,
                    ) = self.user_model._expand_inputs_for_generation(
                        input_ids=torch.vstack(input_ids_padded),
                        expand_size=self.num_beams,
                        is_encoder_decoder=True,
                        **model_kwargs,
                    )

                    # need to recompute these
                    def _get_target_nums(names):
                        for n in names:
                            if hasattr(self.user_model.config, n):
                                return getattr(self.user_model.config, n)
                        logger.error(f"Not found target {names[0]}")
                        exit(1)

                    num_heads_names = [
                        "num_attention_heads",
                        "n_head",
                        "num_heads",
                        "n_heads",
                    ]
                    num_layers_names = [
                        "num_hidden_layers",
                        "n_layer",
                        "num_layers",
                        "n_layers",
                    ]
                    hidden_size_names = ["hidden_size", "n_embd"]
                    n_heads = _get_target_nums(num_heads_names)
                    n_layers = _get_target_nums(num_layers_names)
                    hidden_size = _get_target_nums(hidden_size_names)
                    head_dim = int(hidden_size / n_heads)

                    # lets get the inputs
                    beam_idx_tmp = torch.zeros(
                        (2048, int(self.batch_size * self.num_beams)), dtype=torch.long
                    ).contiguous()
                    input_bs = int(self.batch_size * self.num_beams)
                    last_hidden_state = example_inputs["encoder_outputs"][
                        "last_hidden_state"
                    ]
                    global_past_key_value = tuple(
                        [
                            (
                                torch.zeros(1, 0, 0, 1, dtype=torch.long).contiguous(),
                                torch.zeros([1, n_heads, 1, head_dim]).contiguous(),
                                torch.zeros([1, n_heads, 1, head_dim]).contiguous(),
                                beam_idx_tmp,
                                torch.zeros(1, 0, 0, 1, dtype=torch.long).contiguous(),
                                self.user_model.decoder.block[i]
                                .layer[1]
                                .EncDecAttention.k(last_hidden_state)
                                .view(input_bs, -1, n_heads, head_dim)
                                .transpose(0, 1),
                                self.user_model.decoder.block[i]
                                .layer[1]
                                .EncDecAttention.v(last_hidden_state)
                                .view(input_bs, -1, n_heads, head_dim)
                                .transpose(0, 1),
                                beam_idx_tmp,
                            )
                            for i in range(n_layers)
                        ]
                    )

                    decoder_input_ids = (
                        torch.zeros(input_bs).to(torch.long).unsqueeze(1)
                    )
                    model_inputs = (
                        decoder_input_ids,
                        torch.vstack(attention_mask_padded),
                        tuple(global_past_key_value),
                        (last_hidden_state,),
                    )
                else:
                    raise RuntimeError(
                        "Your model does not match existing example inputs used in ipex smooth quant, exiting..."
                    )

                # Some models require extra inputs
                if re.search(
                    "chatglm", self.user_model.config.architectures[0], re.IGNORECASE
                ):
                    extra_inputs = (torch.tensor(True),)
                    model_inputs = model_inputs + extra_inputs

                return (model_inputs, last_ind)

        calib_dataset = load_dataset(self.calib_dataset, split=self.calib_split)
        logger.info(f"Dataset loaded: {calib_dataset}")
        calib_evaluator = Evaluator(
            self.user_model,
            self.example_inputs_mode,
            self.global_past_key_value,
            calib_dataset,
            self.tokenizer,
            batch_size=self.batch_size,
            num_beams=self.num_beams,
            pad_max=int(self.input_tokens)
            if re.search("t5", config.architectures[0], re.IGNORECASE)
            else 512,
        )
        logger.info(f"Evaluator built: {calib_evaluator}")

        self.calib_dataloader = DataLoader(
            calib_evaluator.dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=calib_evaluator.collate_batch,
        )
        logger.info("Dataloader is built successfully!")

        if IPEX_ENABLE:
            """
            Ipex is enabled, we'll use
            (1) weight only quantization if ipex_weight_only_quantization is enabled
            (2) ipex smooth quantization  if ipex_smooth_quantization is enabled
            (3) ipex bfloat16 optimization if neither is quantization is enabled
            (4) throws error if both 1 and 2 are enabled
            """

            def trace_and_export(model):
                example_inputs = self.get_example_inputs()

                with torch.no_grad(), torch.cpu.amp.autocast(
                    enabled=self.amp_enabled, dtype=self.amp_dtype
                ):
                    self_jit = torch.jit.trace(
                        model.eval(), example_inputs, strict=False, check_trace=False
                    )
                    self_jit = torch.jit.freeze(self_jit.eval())

                    self_jit.save(self.quantized_model_path)

            if self.ipex_smooth_quantization and self.ipex_weight_only_quantization:
                logger.error("Can't enable both SQ and WoQ, enable only one of them")
                exit(1)

            # Clear the cache dir if needed
            if self.clear_cache_dir and os.path.exists(self.quantized_model_path):
                os.remove(self.quantized_model_path)

            if os.path.exists(self.quantized_model_path):
                # this skips all the optimizations and goes to end where we load the model
                logger.info(
                    "A previously quantized model is loaded, if you want to re-quantize the model, enable clear_cache_dir on model config file"
                )

            # lets implement the WOQ
            elif self.ipex_weight_only_quantization:
                weight_dtype = (
                    torch.quint4x2 if self.woq_dtype == "INT4" else torch.qint8
                )

                if self.lowp_mode == "INT8":
                    lowp_mode = ipex.quantization.WoqLowpMode.INT8
                elif self.lowp_mode == "FP32":
                    lowp_mode = ipex.quantization.WoqLowpMode.NONE
                elif self.lowp_mode == "FP16":
                    lowp_mode = ipex.quantization.WoqLowpMode.FP16
                elif self.lowp_mode == "BF16":
                    lowp_mode = ipex.quantization.WoqLowpMode.BF16
                else:
                    lowp_mode = ipex.quantization.WoqLowpMode.BF16

                act_quant_mode_dict = {
                    "PER_TENSOR": ipex.quantization.WoqActQuantMode.PER_TENSOR,
                    "PER_IC_BLOCK": ipex.quantization.WoqActQuantMode.PER_IC_BLOCK,
                    "PER_BATCH": ipex.quantization.WoqActQuantMode.PER_BATCH,
                    "PER_BATCH_IC_BLOCK": ipex.quantization.WoqActQuantMode.PER_BATCH_IC_BLOCK,
                }

                qconfig = ipex.quantization.get_weight_only_quant_qconfig_mapping(
                    weight_dtype=weight_dtype,
                    lowp_mode=lowp_mode,
                    act_quant_mode=act_quant_mode_dict[self.act_quant_mode],
                    group_size=self.group_size,
                )

                # low precision checkpoint can be loaded, but we're considering there isn't any
                low_precision_checkpoint = None
                self.user_model = ipex.llm.optimize(
                    self.user_model.eval(),
                    dtype=self.amp_dtype,
                    quantization_config=qconfig,
                    inplace=True,
                    low_precision_checkpoint=low_precision_checkpoint,
                    deployment_mode=False,
                )
                logger.info(
                    "The model conversion completed, now tracing the quantized model"
                )

                trace_and_export(self.user_model)

                logger.info(
                    "The IPEX Weight only quantization has been completed successfully"
                )

            elif self.ipex_smooth_quantization:
                from intel_extension_for_pytorch.quantization import convert, prepare

                example_inputs = self.get_example_inputs()
                qconfig = ipex.quantization.get_smooth_quant_qconfig_mapping(
                    alpha=self.alpha
                )
                user_model = ipex.llm.optimize(
                    self.user_model.eval(),
                    dtype=self.amp_dtype,
                    quantization_config=qconfig,
                    inplace=True,
                    deployment_mode=False,
                )

                prepared_model = prepare(
                    user_model.eval(),
                    qconfig,
                    example_inputs=example_inputs,
                    inplace=True,
                )
                logger.info("Model prepared for quantization, observers inserted")

                for i, (model_inputs, last_ind) in enumerate(self.calib_dataloader):
                    if i == self.num_calib_iters:
                        break
                    prepared_model(*model_inputs)
                logger.info("Model calibration completed")

                converted_model = convert(prepared_model.eval(), inplace=True).eval()
                logger.info("Model converted successfully, exporting the trace")

                trace_and_export(converted_model)

                logger.info("IPEX Smooth Quantization has completed successfully")

            else:
                # run bf16 model
                example_inputs = self.get_example_inputs()
                self.user_model = ipex.llm.optimize(
                    self.user_model.eval(),
                    dtype=self.amp_dtype,
                    inplace=True,
                    deployment_mode=False,
                )

                trace_and_export(self.user_model)
                logger.info("IPEX bf16 optimization is applied successfully")

            logger.info("Loading the IPEX quantized model")
            try:
                self_jit = torch.jit.load(self.quantized_model_path)
                self_jit = torch.jit.freeze(self_jit.eval())
            except Exception as e:
                logger.error("Error: loading the quantized  model failed.", e)
                exit(0)

            setattr(self.user_model, "trace_graph", self_jit)
            logger.info(
                f"Successfully loaded the Model {model_name} with Intel® Extension for PyTorch*"
            )

        else:
            # No optimization is applied, but if amx is enabled, it'll be applied during generation routine
            logger.warning(
                "No IPEX optimization is applied, Pytorch default autocast will be applied if enabled"
            )

        # set PAD token
        if self.tokenizer.pad_token is None:
            if re.search(
                "qwen", self.user_model.config.architectures[0], re.IGNORECASE
            ):
                self.tokenizer.pad_token = "<|endoftext|>"
            else:
                self.tokenizer.pad_token = self.tokenizer.eos_token

    # we are going to use data collator we built to generate example input
    def get_example_inputs(self):
        (model_inputs, last_ind) = next(iter(self.calib_dataloader))
        return model_inputs

    def preprocess(self, requests):
        input_ids_batch = None
        attention_mask_batch = None
        for idx, data in enumerate(requests):
            input_text = data.get("data")
            if input_text is None:
                input_text = data.get("body")
            if isinstance(input_text, (bytes, bytearray)):
                input_text = input_text.decode("utf-8")

            with torch.inference_mode(), torch.no_grad(), torch.autocast(
                device_type="cpu", enabled=self.amp_enabled, dtype=self.amp_dtype
            ):
                inputs = self.tokenizer(
                    input_text,
                    pad_to_max_length=True,
                    add_special_tokens=True,
                    return_tensors="pt",
                    # max_length=int(self.max_length),
                )

            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            # making a batch out of the recieved requests
            if input_ids.shape is not None:
                if input_ids_batch is None:
                    input_ids_batch = input_ids
                    attention_mask_batch = attention_mask
                else:
                    input_ids_batch = torch.cat((input_ids_batch, input_ids), 0)
                    attention_mask_batch = torch.cat(
                        (attention_mask_batch, attention_mask), 0
                    )
        return (input_ids_batch, attention_mask_batch)

    def inference(self, input_batch):
        input_ids_batch, attention_mask_batch = input_batch
        inferences = []
        # total_list = []

        with torch.inference_mode(), torch.no_grad(), torch.autocast(
            device_type="cpu", enabled=self.amp_enabled, dtype=self.amp_dtype
        ):
            outputs = self.user_model.generate(
                input_ids_batch,
                attention_mask=attention_mask_batch,
                **self.generate_kwargs,
            )
            inferences = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            # for i, x in enumerate(outputs):
            #    inferences.append(self.tokenizer.decode(outputs[i], skip_special_tokens=True))

        return inferences

    def postprocess(self, inference_output):
        return inference_output
