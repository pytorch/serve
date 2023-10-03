import os
import logging
from abc import ABC

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from ts.context import Context
from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)
logger.info("Transformers version %s", transformers.__version__)

IPEX_ENABLE = False
if os.environ.get("TS_IPEX_ENABLE", "false") == "true":
    try:
        import intel_extension_for_pytorch as ipex
        try:
            ipex._C.disable_jit_linear_repack()
        except Exception:
            pass
        IPEX_ENABLE = True
    except ImportError as error:
        logger.warning(
            "IPEX is enabled but intel-extension-for-pytorch is not installed. Proceeding without IPEX."
        )
        IPEX_ENABLE = False

class CodeGenHandler(BaseHandler, ABC):

    def __init__(self):
        super(CodeGenHandler, self).__init__()

    def initialize(self, ctx: Context):
        model_name = ctx.model_yaml_config["handler"]["model_name"]
        model_path = ctx.model_yaml_config["handler"]["model_path"]
        
        self.batch_size = int(ctx.model_yaml_config["handler"]["batch_size"])
        self.max_length = int(ctx.model_yaml_config["handler"]["max_length"])
        self.max_new_tokens = int(ctx.model_yaml_config["handler"]["max_new_tokens"])
        self.min_new_tokens = int(ctx.model_yaml_config["handler"]["min_new_tokens"])
        
        if IPEX_ENABLE:
            ############ Intel® Extension for PyTorch* BF16 JIT ############
            
            # generate args
            num_beams = 4
            self.generate_kwargs = dict(do_sample=False, temperature=0.9, num_beams=num_beams, max_new_tokens=self.max_new_tokens, min_new_tokens=self.max_new_tokens)
            
            # device 
            device = torch.device("cpu")
            
            # jit
            torch._C._jit_set_texpr_fuser_enabled(False)
            
            # load model
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, torchscript=True, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(model_path, torchscript=True, trust_remote_code=True)
            
            self.model = self.model.eval().to(device)
            self.model = self.model.to(memory_format=torch.channels_last)
            
            # to ipex
            self.model = ipex._optimize_transformers(self.model.eval(), dtype=torch.bfloat16, inplace=True)
            
            # dummy past key values
            beam_idx_tmp = torch.zeros(
                (2048, int(self.batch_size * num_beams)), dtype=torch.long
            ).contiguous()
            past_key_values = tuple(
                [
                    (
                        torch.zeros([1, 1, 1, 1]).contiguous(),
                        torch.zeros([1, 1, 1, 1]).contiguous(),
                        beam_idx_tmp,
                        torch.zeros(1, dtype=torch.long).contiguous(),
                    )
                    for i in range(self.model.config.n_layer)
                ]
            )
            
            if not hasattr(self.model, "trace_graph"): 
                input_ids = torch.ones(32).to(torch.long)
                attention_mask = torch.ones(len(input_ids))
                position_ids = torch.arange(len(input_ids))
            
                example_inputs = {
                    "input_ids": input_ids.unsqueeze(0),
                    "attention_mask": attention_mask.unsqueeze(0),
                    "position_ids": position_ids.unsqueeze(0),
                    "past_key_values": past_key_values,
                }
                
                with torch.inference_mode(), torch.no_grad(), torch.autocast(
                    device_type="cpu",
                    enabled=True,
                    dtype=torch.bfloat16
                ):
                    trace_model = torch.jit.trace(
                        self.model, example_kwarg_inputs=example_inputs, strict=False, check_trace=False
                    )
                    trace_model = torch.jit.freeze(trace_model)
                    setattr(self.model, "trace_graph", trace_model)
            logger.info("Successfully optimzied Model %s with Intel® Extension for PyTorch*", ctx.model_name)
            ################################################################
        else:
            # generate args
            self.generate_kwargs = dict(max_new_tokens=self.max_new_tokens, min_new_tokens=self.max_new_tokens)
            
            # load model
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(model_path)
        
        
        # set PAD token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token=self.tokenizer.eos_token
            
        logger.info("Successfully loaded Model %s", ctx.model_name)

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
                enabled=True if IPEX_ENABLE else False,
                dtype=torch.bfloat16 if IPEX_ENABLE else None
            ):
                inputs = self.tokenizer(
                                        input_text,
                                        max_length=int(self.max_length),
                                        pad_to_max_length=True,
                                        add_special_tokens=True,
                                        return_tensors="pt",
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
        
        """
        ############ benchmark ############
        import time
        total_time = 0.0
        num_iter = 100
        num_warmup = 10
        
        prompt = "# This Python script demonstrates a basic Multi-Layer Perceptron (MLP) model for image classification. Using PyTorch machine-learning framework library, it defines a simple MLP architecture, loads the datasets, preprocesses the input images, postprocesses the outputs, and trains it on the training data images. Finally, it evaluates the model's performance on the evaluation data images."
        input_size = self.tokenizer(prompt, return_tensors="pt").input_ids.size(dim=1)
        logger.info("Prompt size: %i", input_size)
        
        with torch.inference_mode(), torch.no_grad(), torch.autocast(
            device_type="cpu",
            enabled=True if IPEX_ENABLE else False,
            dtype=torch.bfloat16 if IPEX_ENABLE else None
        ):
            for i in range(100):
                tic = time.time()
                input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
                gen_ids = self.model.generate(input_ids, **self.generate_kwargs)
                gen_text = self.tokenizer.decode(gen_ids[0], skip_special_tokens=True)
                toc = time.time()
                
                print("Iteration: %d, Time: %.6f sec" % (i, toc - tic), flush=True)
                if i >= num_warmup:
                    total_time += toc - tic
        print("\n", "-" * 10, "Summary:", "-" * 10)
        latency = total_time / (num_iter - num_warmup)
        print("Inference latency: %.3f sec." % latency)
        ###################################
        """
        
        with torch.inference_mode(), torch.no_grad(), torch.autocast(
            device_type="cpu",
            enabled=IPEX_ENABLE==True,
            dtype=torch.bfloat16 if IPEX_ENABLE else None
        ):
            outputs = self.model.generate(input_ids_batch, attention_mask=attention_mask_batch, **self.generate_kwargs)
            for i, x in enumerate(outputs):
                inferences.append(
                    self.tokenizer.decode(outputs[i], skip_special_tokens=True)
                )
        return inferences

    def postprocess(self, inference_output):
        return inference_output