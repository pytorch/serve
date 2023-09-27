import logging
import time
from abc import ABC
import json
import os
import sys
import importlib.util

import packaging.version
import requests
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from ts.torch_handler.base_handler import BaseHandler
current_working_directory = os.getcwd()
sys.path.insert(0,current_working_directory)
from llama2 import Llama
from generate import chat_completion, text_completion, Dialog

logger = logging.getLogger(__name__)
logger.info("Transformers version %s", transformers.__version__)
if packaging.version.parse(torch.__version__) >= packaging.version.parse("2.0.0"):
    logger.info("PyTorch version is 2.0.0 or greater")
else:
    logger.info(
        "PyTorch version is less than 2.0.0, initializing with meta device needs PyTorch 2.0.0 and greater"
    )



class LlamaHandler(BaseHandler,ABC):
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

        self.manifest = ctx.manifest
        properties = ctx.system_properties
        model_dir = properties.get("model_dir")
        
        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id"))
            if torch.cuda.is_available() and properties.get("gpu_id") is not None
            else "cpu"
        )

        seed = ctx.model_yaml_config["handler"]["manual_seed"]
        self.mode = ctx.model_yaml_config["handler"]["mode"]
        self.max_new_tokens = ctx.model_yaml_config["handler"]["max_new_tokens"]
        self.temperature = ctx.model_yaml_config["handler"]["temperature"]
        self.top_p = ctx.model_yaml_config["handler"]["top_p"]
        
        torch.manual_seed(seed)

        logger.info("Instantiating Llama model")
        model_load_start = time.perf_counter()
        llama_model_and_tok=  Llama.build(
            model_args=ctx.model_yaml_config["handler"]["model_args_path"],
            converted_ckpt_dir=ctx.model_yaml_config["handler"]["converted_ckpt_dir"],
            tokenizer_path= ctx.model_yaml_config["handler"]["tokenizer_path"],
        )
        load_time = time.perf_counter()-model_load_start
        self.model = llama_model_and_tok.model
    
  
        self.tokenizer = llama_model_and_tok.tokenizer

        logger.info(f"Llama model from path {model_dir} loaded successfully in {load_time} seconds")

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
        input_texts = [data.get("data") or data.get("body") for data in requests]
        if self.mode == "chat":
            try:
                dialogs_list = [json.loads(text) for text in input_texts if text]
                return dialogs_list
            except json.JSONDecodeError:
                print("Error: Unable to decode JSON. Please ensure the input text is correctly formatted.")
            

        elif self.mode == "text_completion":
            
            try:
                return [self.prep_input_text(text) for text in input_texts]
            except Exception as e:
                print(f"Error while preparing input text: {e}")
            # try:
            #     input_ids_batch = []
            #     for input_text in input_texts:
            #         input_ids = self.prep_input_text(input_text)
            #         input_ids_batch.append(input_ids)
            #     return input_ids_batch
            # except Exception as e:
            #     print(f"Error while preparing input text: {e}")
            
        else:
            print("Error: Unsupported mode. Please select a valid mode.")
            return []

    def prep_input_text(self, input_text):
        """
        preparing a single input text using the tokenizer.
        Args:
            input_text (str): The input text to be encoded.
        Returns:
            decoded input text
        """
        if isinstance(input_text, (bytes, bytearray)):
            input_text = input_text.decode("utf-8")
        logger.info("Received text: '%s'", input_text)
        
        return input_text

    def inference(self, input_batch):
        """
        Generate tokens based on prompts
        Args:
            input_batch : a batch of input texts
        Returns:
            list: A list of strings with the predicted values for each input text in the batch.
        """
        
        print("in the inference call")
        if self.mode == "chat":
            
            for dialog in input_batch:
                results = chat_completion(
                        self.model,
                        self.tokenizer,
                        dialog,
                        max_gen_len=self.max_new_tokens,
                        temperature=self.temperature,
                        top_p=self.top_p,
                    )
            
        elif self.mode == "text_completion":
            results = text_completion(
                    self.model,
                    self.tokenizer,
                    input_batch,
                    max_gen_len=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                )
      
        
        
        return results

    def postprocess(self, inference_output):
        """Post Process Function converts the predicted response into Torchserve readable format.
        Args:
            inference_output (list): It contains the predicted response of the input text.
        Returns:
            (list): Returns a list of the Predictions and Explanations.
        """
        
        logger.info("Generated text: %s", inference_output)
        
        return [inference_output]
