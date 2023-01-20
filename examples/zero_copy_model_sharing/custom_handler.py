import json
import logging
import os
from abc import ABC

import torch
import transformers
from transformers import BloomForCausalLM, BloomTokenizerFast

from ts.shared_memory_model_store import SharedMemoryModelStore
from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)
logger.info("Transformers version %s", transformers.__version__)

torch.set_num_threads(1)


TORCH_DTYPES = {
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
}


class ZeroCopyModelSharingHandler(BaseHandler, ABC):
    """
    This handler shares the model with other workers
    The first worker loads the model and moves it into shared memory
    The next worker only looks up the model in a common model store
    """

    def __init__(self):
        super(ZeroCopyModelSharingHandler, self).__init__()
        self.initialized = False

    def initialize(self, ctx):
        """In this initialize function, the worker determines if its the first worker
        In this case it create the model, moves it to shared memory and put its into a store
        If the worker is not the first it only opens the store and loads the model from there
        Args:
            ctx (context): It is a JSON Object containing information
            pertaining to the model artefacts parameters.
        """
        self.manifest = ctx.manifest
        properties = ctx.system_properties
        model_dir = properties.get("model_dir")

        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id"))
            if torch.cuda.is_available() and properties.get("gpu_id") is not None
            else "cpu"
        )
        # Loading the model, tokenizer and read configs for the mode, model_name, etc. from setup_config.json
        setup_config_path = os.path.join(model_dir, "setup_config.json")
        if os.path.isfile(setup_config_path):
            with open(setup_config_path) as setup_config_file:
                self.setup_config = json.load(setup_config_file)
        else:
            logger.warning("Missing the setup_config.json file.")

        identifier = f"{self.manifest['model']['modelName']}_{self.manifest['model']['modelVersion']}"

        self.store = SharedMemoryModelStore(identifier)

        if self.store.is_master:
            model = BloomForCausalLM.from_pretrained(
                self.setup_config["model_name"],
                torch_dtype=TORCH_DTYPES[self.setup_config["torch_dtype"]],
            )

            model = self.store.set(
                self.setup_config["model_name"].replace("/", ""), model
            )
        else:
            model = self.store.get(self.setup_config["model_name"].replace("/", ""))

        self.model = model

        self.tokenizer = BloomTokenizerFast.from_pretrained(
            self.setup_config["model_name"], return_tensors="pt"
        )

        self.model.to(self.device)
        self.model.eval()
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
        for data in requests:
            input_text = data.get("data")
            if input_text is None:
                input_text = data.get("body")
            if isinstance(input_text, (bytes, bytearray)):
                input_text = input_text.decode("utf-8")

            input_text = input_text if input_text != "" else " "

            max_length = self.setup_config["max_length"]
            logger.info("Received text: '%s'", input_text)

            inputs = self.tokenizer.encode_plus(
                input_text,
                max_length=int(max_length),
                truncation=True,
                padding=True,
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
        outputs = self.model.generate(
            input_ids_batch, do_sample=True, max_length=50, top_p=0.95, top_k=60
        )
        for i, _ in enumerate(outputs):
            inferences.append(
                self.tokenizer.decode(outputs[i], skip_special_tokens=True)
            )

        logger.info("Generated text: '%s'", inferences)

        return inferences

    def postprocess(self, inference_output):
        """Post Process Function converts the predicted response into Torchserve readable format.
        Args:
            inference_output (list): It contains the predicted response of the input text.
        Returns:
            (list): Returns a list of the Predictions and Explanations.
        """
        return inference_output
