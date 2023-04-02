import json
import logging
import os
import zipfile
from abc import ABC

import torch
import transformers
from transformers import BloomForCausalLM, BloomTokenizerFast, AutoModelForCausalLM, AutoTokenizer

from ts.torch_handler.distributed.base_pippy_handler import BasePippyHandler
import argparse
import inspect
import logging
import os
import time

import torch
from PIL import Image
import requests
import torch.distributed.rpc as rpc
from ts.handler_utils.distributed.pt_pippy import get_pipline_driver

logger = logging.getLogger(__name__)
logger.info("Transformers version %s", transformers.__version__)


class TransformersSeqClassifierHandler(BasePippyHandler, ABC):
    """
    Transformers handler class for sequence, token classification and question answering.
    """

    def __init__(self):
        super(TransformersSeqClassifierHandler, self).__init__()
        self.initialized = False
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])

    def initialize(self, ctx):
        """In this initialize function, the HF large model is loaded and 
        partitioned into multiple stages each on one device using PiPPy.
        Args:
            ctx (context): It is a JSON Object containing information
            pertaining to the model artefacts parameters.
        """
   

        self.manifest = ctx.manifest
        properties = ctx.system_properties
        model_dir = properties.get("model_dir")
        n_devs = torch.cuda.device_count()
        self.device = self.local_rank % n_devs
        with zipfile.ZipFile(model_dir + "/model.zip", "r") as zip_ref:
            zip_ref.extractall(model_dir + "/model")

        torch.manual_seed(42)
     

        model = AutoModelForCausalLM.from_pretrained(
            model_dir + "/model", use_cache=False)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir + "/model", return_tensors="pt"
        )
        
        model.eval()

        chunks = ctx.model_yaml_config["chunks"]
        input_names = ctx.model_yaml_config["input_names"]
        model_type= ctx.model_yaml_config["model_type"]
     

        print('Instantiating model Pipeline')
        model_init_start = time.time()
        self.model  = get_pipline_driver(model,self.world_size, input_names, model_type, chunks)

        logger.info("Transformer model from path %s loaded successfully", model_dir)

        self.initialized = True


    def preprocess(self, requests):
        """Basic text preprocessing, based on the user's chocie of application mode.
        Args:
            requests (str): The Input data in the form of text is passed on to the preprocess
            function.
        Returns:
            list : The preprocess function returns a list of Tensor for the size of the word tokens.
        """
        input_ids_batch = None
        attention_mask_batch = None
        for idx, data in enumerate(requests):
            input_text = data.get("data")
            if input_text is None:
                input_text = data.get("body")
            if isinstance(input_text, (bytes, bytearray)):
                input_text = input_text.decode("utf-8")

            max_length = self.setup_config["max_length"]
            logger.info("Received text: '%s'", input_text)

            inputs = self.tokenizer.encode_plus(
                input_text,
                max_length=int(max_length),
                pad_to_max_length=True,
                add_special_tokens=True,
                return_tensors="pt",
            )

            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)
            # making a batch out of the recieved requests
            # attention masks are passed for cases where input tokens are padded.
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
        """Predict the class (or classes) of the received text using the
        serialized transformers checkpoint.
        Args:
            input_batch (list): List of Text Tensors from the pre-process function is passed here
        Returns:
            list : It returns a list of the predicted value for the input text
        """
        (input_ids_batch, _) = input_batch
        inferences = []
        input_ids_batch = input_ids_batch.to(self.device)
        model_input_dict = {}
        model_input_dict["input_ids"]=input_ids_batch
        # outputs = self.model.generate(
        #     input_ids_batch, do_sample=True, max_length=50, top_p=0.95, top_k=60
        # )
        # for i, _ in enumerate(outputs):
        #     inferences.append(
        #         self.tokenizer.decode(outputs[i], skip_special_tokens=True)
        #     )
        if self.local_rank==0:
            output = self.model(**model_input_dict)
        # rpc.shutdown()
        print("************** here is the output",type(output["logits"]), len(output["logits"]))
        # print(self.tokenizer.decode(output["logits"], skip_special_tokens=True))
        # inference = self.tokenizer.decode(output["logits"], skip_special_tokens=True)
        # logger.info("Generated text: '%s'", inferences)
        inferences.append("inference")
        print("Generated text", inferences)
        return inferences

    def postprocess(self, inference_output):
        """Post Process Function converts the predicted response into Torchserve readable format.
        Args:
            inference_output (list): It contains the predicted response of the input text.
        Returns:
            (list): Returns a list of the Predictions and Explanations.
        """
        return inference_output

   
