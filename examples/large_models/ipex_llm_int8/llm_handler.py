import os
import logging
from abc import ABC
from pathlib import Path
import subprocess

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from datasets import load_dataset
from torch.utils.data import DataLoader

from ts.context import Context
from ts.torch_handler.base_handler import BaseHandler
import intel_extension_for_pytorch as ipex


EXAMPLE_INPUTS_MODE = {
        "MASK_KV": 1,
        "KV_MASK": 2,
        "MASK_POS_KV": 3,
        "MASK_KV_POS": 4,
        "MASK_KV_ENC": 5,
}


logger = logging.getLogger(__name__)
logger.info("PyTorch version %s", torch.__version__)
logger.info("IPEX version %s", ipex.__version__)
logger.info("Transformers version %s", transformers.__version__)

class IpexLLMHandler(BaseHandler, ABC):

    def __init__(self):
        super(IpexLLMHandler, self).__init__()
        
        # for streaming the generated texts back to client
        self.output_streamer = None


    def initialize(self, ctx: Context):
        model_name = ctx.model_yaml_config["handler"]["model_name"]
        # path to quantized model, if we are quantizing on the fly, we'll use this path to save the model
        self.quantized_model_path = ctx.model_yaml_config["handler"]["quantized_model_path"]
        self.example_inputs_mode = ctx.model_yaml_config["handler"]["example_inputs_mode"]
        self.to_channels_last = ctx.model_yaml_config["handler"]["to_channels_last"]
        
        # generation params
        self.batch_size = int(ctx.model_yaml_config["handler"]["batch_size"])
        self.max_context_length = int(ctx.model_yaml_config["handler"]["max_context_length"])
        self.input_tokens = int(ctx.model_yaml_config["handler"]["input_tokens"])
        self.max_new_tokens = int(ctx.model_yaml_config["handler"]["max_new_tokens"])
        
        # use int8 bf16 mix
        self.quant_with_amp = ctx.model_yaml_config["handler"]["quant_with_amp"]
        
        # WoQ related optimization params 
        if "ipex_weight_only_quantization" in ctx.model_yaml_config["handler"]:
            self.ipex_weight_only_quantization = ctx.model_yaml_config["handler"]["ipex_weight_only_quantization"]
            self.woq_dtype = ctx.model_yaml_config["handler"]["woq_dtype"]
            self.lowp_mode = ctx.model_yaml_config["handler"]["lowp_mode"]
            self.act_quant_mode = ctx.model_yaml_config["handler"]["act_quant_mode"] # This is only relevant for INT4x2 quantization
            self.group_size = ctx.model_yaml_config["handler"]["group_size"]
        else:
            self.ipex_weight_only_quantization = False

        # SQ related optimization params 
        if "ipex_smooth_quantization" in ctx.model_yaml_config["handler"]:
            self.ipex_smooth_quantization = ctx.model_yaml_config["handler"]["ipex_smooth_quantization"]
            self.calib_dataset = ctx.model_yaml_config["handler"]["calibration_dataset"]
            self.calib_split = ctx.model_yaml_config["handler"]["calibration_split"]
            self.num_calib_iters = int(ctx.model_yaml_config["handler"]["num_calibration_iters"])
            self.alpha = float(ctx.model_yaml_config["handler"]["alpha"])
        else:
            self.ipex_smooth_quantization = False
        

        # decoding parameters 
        self.greedy = ctx.model_yaml_config["handler"]["greedy"]
        logger.info(f"Max length of the sequence context is {self.max_context_length}")

        try:
            ipex._C.disable_jit_linear_repack()
            torch._C._jit_set_texpr_fuser_enabled(False)
        except Exception:
            pass

        # amp datatype 
        if self.quant_with_amp: 
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
            min_new_tokens=self.max_new_tokens,
        ) 
        
        # device 
        device = torch.device("cpu")
        
        # model config 
        config = AutoConfig.from_pretrained(model_name, torchscript=True, trust_remote_code=True)
        
        # set up max context 
        if not hasattr(config, "text_max_length"):
            config.text_max_length = int(self.max_context_length)
        
        # load model and tokenizer
        self.user_model = AutoModelForCausalLM.from_pretrained(model_name, config=config, low_cpu_mem_usage=True, torch_dtype=torch.float)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code = True)

        logger.info("Data type of the model: %s", self.user_model.dtype)
        
        if self.to_channels_last:
            self.user_model = self.user_model.to(memory_format=torch.channels_last)
        self.user_model.eval()
        

        # dummy past key value
        beam_idx_tmp = torch.zeros((2048, int(self.batch_size * self.num_beams)), dtype=torch.long).contiguous()
        def _get_target_nums(names):
            for n in names:
                if hasattr(self.user_model.config, n):
                    return getattr(self.user_model.config, n)
            logger.error(f"Not found target {names[0]}")
            exit(0)

        num_heads_names = ["num_attention_heads", "n_head", "num_heads", "n_heads"]
        num_layers_names = ["num_hidden_layers", "n_layer", "num_layers", "n_layers"]
        hidden_size_names = ["hidden_size", "n_embd"]
        n_heads = _get_target_nums(num_heads_names)
        n_layers = _get_target_nums(num_layers_names)
        hidden_size = _get_target_nums(hidden_size_names)
        head_dim = int(hidden_size / n_heads)
        self.global_past_key_value = [
            (
                torch.zeros(1, 0, 0, 1, dtype=torch.long).contiguous(),
                torch.zeros([1, n_heads, 1, head_dim]).contiguous(),
                torch.zeros([1, n_heads, 1, head_dim]).contiguous(),
                beam_idx_tmp,
            )
            for i in range(n_layers)
        ]

        logger.info(f"num_attention_heads: {n_heads}, num_hidden_layers: {n_layers}, hidden size: {hidden_size}, head_dim: {head_dim}")

        if self.ipex_smooth_quantization and self.ipex_weight_only_quantization:
            logger.error("Can't enable both SQ and WoQ, enable only one of them")
            exit(1)

        # lets implement the WOQ 
        if self.ipex_weight_only_quantization:
            weight_dtype = torch.quint4x2 if self.woq_dtype == "INT4" else torch.qint8

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
            logger.info("The model conversion completed, now tracing the quantized model")

            example_inputs = self.get_example_inputs()

            with torch.no_grad(), torch.cpu.amp.autocast(
                enabled=self.amp_enabled,
                dtype=self.amp_dtype
            ):
                self_jit = torch.jit.trace(self.user_model.eval(), example_inputs, strict=False, check_trace=False)
                self_jit = torch.jit.freeze(self_jit.eval())

                self_jit.save(self.quantized_model_path)

            logger.info("The IPEX Weight only quantization has been completed successfully")

        elif self.ipex_smooth_quantization:
            class Evaluator:
                def __init__(self, example_inputs_mode, global_past_key_value, dataset, tokenizer, batch_size=1, num_beams=1, pad_val=1, pad_max=512):
                    self.example_inputs_mode = example_inputs_mode
                    self.global_past_key_value = global_past_key_value
                    self.dataset = dataset
                    self.tokenizer = tokenizer
                    self.batch_size = batch_size
                    self.num_beams = num_beams


                    self.pad_val = pad_val
                    self.pad_max = pad_max 
                    self.dataset = self.dataset.map(self.tokenize_function, batched = True, num_proc=2)
                    self.dataset.set_format(type="torch", columns=["input_ids"])  

                @torch.no_grad()
                def tokenize_function(self, examples):
                    if "prompt" in examples:
                        example = self.tokenizer(examples["prompt"])
                    elif "text" in examples:
                        example = self.tokenizer(examples["text"])
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
                        model_kwargs = user_model._prepare_encoder_decoder_kwargs_for_generation(
                            torch.vstack(input_ids_padded), model_kwargs, "input_ids"
                        )
                        input_ids, example_inputs = user_model._expand_inputs_for_generation(
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
                            exit(0)

                        num_heads_names = ["num_attention_heads", "n_head", "num_heads", "n_heads"]
                        num_layers_names = ["num_hidden_layers", "n_layer", "num_layers", "n_layers"]
                        hidden_size_names = ["hidden_size", "n_embd"]
                        n_heads = _get_target_nums(num_heads_names)
                        n_layers = _get_target_nums(num_layers_names)
                        hidden_size = _get_target_nums(hidden_size_names)
                        head_dim = int(hidden_size / n_heads)
                        
                        # lets get the inputs
                        input_bs = int(self.batch_size * self.num_beams)
                        last_hidden_state = example_inputs["encoder_outputs"]["last_hidden_state"]
                        global_past_key_value = tuple(
                            [
                                (
                                    torch.zeros(1, 0, 0, 1, dtype=torch.long).contiguous(),
                                    torch.zeros([1, n_heads, 1, head_dim]).contiguous(),
                                    torch.zeros([1, n_heads, 1, head_dim]).contiguous(),
                                    beam_idx_tmp,
                                    torch.zeros(1, 0, 0, 1, dtype=torch.long).contiguous(),
                                    user_model.decoder.block[i].layer[1].EncDecAttention.k(last_hidden_state).view(input_bs, -1, n_heads, head_dim).transpose(0, 1),
                                    user_model.decoder.block[i].layer[1].EncDecAttention.v(last_hidden_state).view(input_bs, -1, n_heads, head_dim).transpose(0, 1),
                                    beam_idx_tmp,
                                )
                                for i in range(n_layers)
                            ]
                        )

                        decoder_input_ids = (torch.zeros(input_bs).to(torch.long).unsqueeze(1))
                        model_inputs = (
                            decoder_input_ids,
                            torch.vstack(attention_mask_padded),
                            tuple(global_past_key_value),
                            (last_hidden_state,),
                        )
                    else:
                        raise RuntimeError("Your model does not match existing example inputs used in ipex smooth quant, exiting...")

                    return (model_inputs, last_ind)



            calib_dataset = load_dataset(self.calib_dataset, split=self.calib_split)
            logger.info(f"Dataset loaded: {calib_dataset}")
            calib_evaluator = Evaluator(
                self.example_inputs_mode,
                self.global_past_key_value,
                calib_dataset, 
                self.tokenizer, 
                batch_size=self.batch_size, 
                num_beams = self.num_beams, 
                pad_max = 512
            )
            logger.info(f"Evaluator built: {calib_evaluator}")

            calib_dataloader = DataLoader(
                calib_evaluator.dataset,
                batch_size=1,
                shuffle=False,
                collate_fn=calib_evaluator.collate_batch,
            )
            logger.info("Dataloader ready")


            from intel_extension_for_pytorch.quantization import prepare, convert
            example_inputs = self.get_example_inputs()
            qconfig = ipex.quantization.get_smooth_quant_qconfig_mapping(alpha=self.alpha)
            user_model = ipex.llm.optimize(
                self.user_model.eval(),
                dtype=self.amp_dtype,
                quantization_config=qconfig,
                inplace=True,
                deployment_mode=False,
            )

            prepared_model = prepare(user_model.eval(), qconfig, example_inputs=example_inputs, inplace=True)
            logger.info("Model prepared for quantization, observers inserted")


            for i, (model_inputs, last_ind) in enumerate(calib_dataloader):
                if i == self.num_calib_iters:
                    break
                prepared_model(*model_inputs)
            logger.info("Model calibration completed")

            converted_model = convert(prepared_model.eval(), inplace=True).eval()
            logger.info("Model converted successfully, exporting the trace")
            
            with torch.no_grad(), torch.cpu.amp.autocast(
                enabled=self.amp_enabled,
                dtype=self.amp_dtype
            ):
                self_jit = torch.jit.trace(converted_model.eval(), example_inputs, strict=False, check_trace=False)
                self_jit = torch.jit.freeze(self_jit.eval())

                self_jit.save(self.quantized_model_path)

            logger.info("IPEX Smooth Quantization has completed successfully")

        else:
            # run bf16 model
            example_inputs = self.get_example_inputs()
            self.user_model = ipex.llm.optimize(
                self.user_model.eval(),
                dtype = self.amp_dtype,
                inplace=True,
                deployment_mode=False,
            )
            
            with torch.no_grad(), torch.cpu.amp.autocast(
                enabled=self.amp_enabled,
                dtype=self.amp_dtype
            ):
                self_jit = torch.jit.trace(self.user_model.eval(), example_inputs, strict=False, check_trace=False)
                self_jit = torch.jit.freeze(self_jit.eval())

                self_jit.save(self.quantized_model_path)

            logger.info("IPEX bf16 optimization is applied successfully")
            
        # set PAD token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token=self.tokenizer.eos_token
            
        logger.info("Loading the IPEX quantized model")
        try:
            self_jit = torch.jit.load(self.quantized_model_path)
            self_jit = torch.jit.freeze(self_jit.eval())
        except Exception as e:
            logger.error("Error: loading the quantized  model failed.", e)
            exit(0)
        
        setattr(self.user_model, "trace_graph", self_jit)
        logger.info("Successfully loaded the Model %s with Intel® Extension for PyTorch*", ctx.model_name)
        
    # Different model need to have their inputs supplied in different order unless we pass dict 
    # For torchserve sending dict is not always possible
    # This function reorders the input ids, masks, and kv cache based on models 
    def get_example_inputs(self):
        example_inputs = None
        input_ids = torch.ones(32).to(torch.long)
        attention_mask = torch.ones(len(input_ids))
        if self.example_inputs_mode == "MASK_POS_KV":
            position_ids = torch.arange(len(input_ids))
            example_inputs = (
                input_ids.unsqueeze(0),
                attention_mask.unsqueeze(0),
                position_ids.unsqueeze(0),
                tuple(self.global_past_key_value),
            )
        elif self.example_inputs_mode == "MASK_KV_POS":
            position_ids = torch.arange(len(input_ids))
            example_inputs = (
                input_ids.unsqueeze(0),
                attention_mask.unsqueeze(0),
                tuple(self.global_past_key_value),
                position_ids.unsqueeze(0),
            )
        elif self.example_inputs_mode == "KV_MASK":
            example_inputs = (
                input_ids.unsqueeze(0),
                tuple(self.global_past_key_value),
                attention_mask.unsqueeze(0),
            )
        elif self.example_inputs_mode == "MASK_KV":
            example_inputs = (
                input_ids.unsqueeze(0),
                attention_mask.unsqueeze(0),
                tuple(self.global_past_key_value),
            )
        elif self.example_inputs_mode == "MASK_KV_ENC":
            last_hidden_state = torch.rand([1, 32, 2048])
            
            #need to recompute these
            def _get_target_nums(names):
                for n in names:
                    if hasattr(self.user_model.config, n):
                        return getattr(self.user_model.config, n)
                logger.error(f"Not found target {names[0]}")
                exit(0)

            num_heads_names = ["num_attention_heads", "n_head", "num_heads", "n_heads"]
            num_layers_names = ["num_hidden_layers", "n_layer", "num_layers", "n_layers"]
            hidden_size_names = ["hidden_size", "n_embd"]
            n_heads = _get_target_nums(num_heads_names)
            n_layers = _get_target_nums(num_layers_names)
            hidden_size = _get_target_nums(hidden_size_names)
            head_dim = int(hidden_size / n_heads)
            
            global_past_key_value = [
                (
                    torch.zeros(1, 0, 0, 1, dtype=torch.long).contiguous(),
                    torch.zeros([1, n_heads, 1, head_dim]).contiguous(),
                    torch.zeros([1, n_heads, 1, head_dim]).contiguous(),
                    beam_idx_tmp,
                    torch.zeros(1, 0, 0, 1, dtype=torch.long).contiguous(),
                    torch.zeros([32, 1, n_heads, head_dim]).contiguous(),
                    torch.zeros([32, 1, n_heads, head_dim]).contiguous(),
                    beam_idx_tmp,
                )
                for i in range(n_layers)
            ]
            example_inputs = (
                torch.ones(1).to(torch.long).unsqueeze(0),
                attention_mask.unsqueeze(0),
                tuple(global_past_key_value),
                (last_hidden_state,),
            )
        else:
            raise RuntimeError("Your model does not match existing example inputs used in ipex quantization, exiting...")
        #if hasattr(model, "extra_inputs"):
        #    example_inputs = example_inputs + model.extra_inputs
        return example_inputs

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
                device_type="cpu",
                enabled=self.amp_enabled,
                dtype=self.amp_dtype
            ):
                inputs = self.tokenizer(
                                        input_text,
                                        pad_to_max_length=True,
                                        add_special_tokens=True,
                                        return_tensors="pt",
                                        #max_length=int(self.max_length),
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
                    attention_mask_batch = torch.cat((attention_mask_batch, attention_mask), 0)
        return (input_ids_batch, attention_mask_batch)
        
    def inference(self, input_batch):
        input_ids_batch, attention_mask_batch = input_batch
        inferences = []
        # total_list = []
        
        with torch.inference_mode(), torch.no_grad(), torch.autocast(
            device_type="cpu",
            enabled=self.amp_enabled,
            dtype=self.amp_dtype
        ):
            outputs = self.user_model.generate(input_ids_batch, attention_mask=attention_mask_batch, **self.generate_kwargs)
            for i, x in enumerate(outputs):
                inferences.append(self.tokenizer.decode(outputs[i], skip_special_tokens=True))

        return inferences

    def postprocess(self, inference_output):
        return inference_output
