import os
import logging
from abc import ABC
from pathlib import Path
import subprocess

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

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
logger.info("Transformers version %s", transformers.__version__)

class CodeGenHandler(BaseHandler, ABC):

    def __init__(self):
        super(CodeGenHandler, self).__init__()

    def initialize(self, ctx: Context):
        model_name = ctx.model_yaml_config["handler"]["model_name"]
        # path to quantized model, if we are quantizing on the fly, we'll use this path to save the model
        self.quantized_model_path = ctx.model_yaml_config["handler"]["quantized_model_path"]
        self.example_inputs_mode = ctx.model_yaml_config["handler"]["example_inputs_mode"]
        self.to_channels_last = ctx.model_yaml_config["handler"]["to_channels_last"]
        
        # generation params
        self.batch_size = int(ctx.model_yaml_config["handler"]["batch_size"])
        self.max_length = int(ctx.model_yaml_config["handler"]["max_length"])

        # when benchmarking, we'll limit both the min and max token to this number for exact measurement 
        self.max_new_tokens = int(ctx.model_yaml_config["handler"]["max_new_tokens"])
        
        # optimization params: right now we're only using WOQ, for SQ and other approach need to add support 
        self.ipex_weight_only_quantization = ctx.model_yaml_config["handler"]["ipex_weight_only_quantization"]
        self.woq_dtype = ctx.model_yaml_config["handler"]["woq_dtype"]
        self.lowp_mode = ctx.model_yaml_config["handler"]["lowp_mode"]
        self.act_quant_mode = ctx.model_yaml_config["handler"]["act_quant_mode"] # This is only relevant for INT4x2 quantization
        self.group_size = ctx.model_yaml_config["handler"]["group_size"]
        
        # runtime params 
        self.benchmark = ctx.model_yaml_config["handler"]["benchmark"]
        self.token_latency = ctx.model_yaml_config["handler"]["token_latency"]
        self.num_warmup = ctx.model_yaml_config["handler"]["num_warmup"]
        self.num_iter = ctx.model_yaml_config["handler"]["num_iter"]
        
        # decoding parameters 
        self.greedy = ctx.model_yaml_config["handler"]["greedy"]
        logger.info(f"Max length of the sequence context is {self.max_length}")

        try:
            ipex._C.disable_jit_linear_repack()
            torch._C._jit_set_texpr_fuser_enabled(False)
        except Exception:
            pass

        # amp datatype 
        if self.lowp_mode == "BF16":
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
            config.text_max_length = int(self.max_length)
        
        # load model and tokenizer
        #model = CodeGen25Config(model_name)
        #self.user_model = model.get_user_model(config, self.benchmark)
        #self.tokenizer = model.get_tokenizer() 
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
            
        # set PAD token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token=self.tokenizer.eos_token
        

        if self.benchmark:
            print("Loading the quantized model")
            # Quantizing the model may be needed before loading the quantized jit 

            try:
                self_jit = torch.jit.load(self.quantized_model_path)
                self_jit = torch.jit.freeze(self_jit.eval())
                print(self_jit)
            except Exception as e:
                logger.error("Error: loading the model failed.", e)
                exit(0)
        
            setattr(self.user_model, "trace_graph", self_jit)
            logger.info("Successfully loaded the Model %s with Intel® Extension for PyTorch*", ctx.model_name)


            if self.token_latency:
                if not hasattr(self.user_model.config, "token_latency"):
                    self.user_model.config.token_latency = True

        
            # set min token count for exact number of token generation    
            benchmark_generate_kwargs = dict(do_sample=False, temperature=0.9, num_beams=self.num_beams, max_new_tokens=self.max_new_tokens, min_new_tokens=self.max_new_tokens)
            import time
            total_time = 0.0
            total_list = []
            num_iter = self.num_iter
            num_warmup = self.num_warmup
            
            input_size = self.max_length - self.max_new_tokens
            input_ids = torch.randint(1, 10000, (self.batch_size, input_size)).to(torch.long)

            logger.info("Prompt size: %i", input_size)
            
            with torch.inference_mode(), torch.no_grad(), torch.autocast(
                device_type="cpu",
                enabled=self.amp_enabled,
                dtype=self.amp_dtype,
            ):
                for i in range(num_iter):
                    tic = time.time()
                    output = self.user_model.generate(input_ids, **benchmark_generate_kwargs)
                    gen_ids = output[0] if self.token_latency else output
                    gen_text = self.tokenizer.decode(gen_ids[0], skip_special_tokens=True)
                    toc = time.time()
                    
                    logger.info("Iteration: %d, Time: %.6f sec" % (i, toc - tic))
                    if i >= num_warmup:
                        total_time += toc - tic
                        if self.token_latency:
                            total_list.append(output[1])

            latency = total_time / (num_iter - num_warmup)
            logger.info("Average Query Latency: %.3f sec." % latency)

            if self.token_latency:
                import numpy as np
                from itertools import chain
                first_latency = np.mean([x[0] for x in total_list])
                average_2n = list(chain(*[x[1:] for x in total_list]))
                average_2n.sort()
                average_2n_latency = np.mean(average_2n)
                p90_latency = average_2n[int(len(average_2n)*0.9)]
                p99_latency = average_2n[int(len(average_2n)*0.99)]

                logger.info("Average first token latency: %.3f sec." % first_latency)
                logger.info("Average second token latency: %.3f sec." % average_2n_latency)
                logger.info("P90 second token latency: %.3f sec." % p90_latency)
                logger.info("P99 second token latency: %.3f sec." % p99_latency)

            self.benchmark = False
            self.user_model.config.token_latency = False

        logger.info("Successfully benchmarked the model, ready to take prompts")
        
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
            logger.info(f"The shape of received batch input ids is {input_ids.shape}")
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
                logger.info(f"The output shape is {outputs[i].shape}")
                inferences.append(self.tokenizer.decode(outputs[i], skip_special_tokens=True))

        return inferences

    def postprocess(self, inference_output):
        return inference_output
