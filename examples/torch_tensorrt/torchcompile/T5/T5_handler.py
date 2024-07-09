import logging

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)


class T5Handler(BaseHandler):
    """
    Transformers handler class for sequence, token classification and question answering.
    """

    def __init__(self):
        super(T5Handler, self).__init__()
        self.tokenizer = None
        self.model = None
        self.initialized = False

    def initialize(self, ctx):
        """In this initialize function, the T5 model is loaded. It also has
           the torch.compile calls for encoder and decoder.
        Args:
            ctx (context): It is a JSON Object containing information
            pertaining to the model artifacts parameters.
        """
        self.manifest = ctx.manifest
        self.model_yaml_config = (
            ctx.model_yaml_config
            if ctx is not None and hasattr(ctx, "model_yaml_config")
            else {}
        )
        properties = ctx.system_properties
        model_dir = properties.get("model_dir")

        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id"))
            if torch.cuda.is_available() and properties.get("gpu_id") is not None
            else "cpu"
        )

        # read configs for the mode, model_name, etc. from the handler config
        model_path = self.model_yaml_config.get("handler", {}).get("model_path", None)
        if not model_path:
            logger.error("Missing model path")

        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)
        self.model.to(self.device)

        self.model.eval()

        pt2_value = self.model_yaml_config.get("pt2", {})
        if "compile" in pt2_value:
            compile_options = pt2_value["compile"]
            if compile_options["enable"] == True:
                del compile_options["enable"]

                compile_options_str = ", ".join(
                    [f"{k} {v}" for k, v in compile_options.items()]
                )
                self.model.encoder = torch.compile(
                    self.model.encoder,
                    **compile_options,
                )
                self.model.decoder = torch.compile(
                    self.model.decoder,
                    **compile_options,
                )
                logger.info(f"Compiled model with {compile_options_str}")
        logger.info("T5 model from path %s loaded successfully", model_dir)

        self.initialized = True

    def preprocess(self, requests):
        """
        Basic text preprocessing, based on the user's choice of application mode.
        Args:
            requests (list): A list of dictionaries with a "data" or "body" field, each
                            containing the input text to be processed.
        Returns:
            inputs: A batched tensor of inputs: the batch of input ids and
                attention masks.
        """

        # Prefix for translation from English to German
        task_prefix = "translate English to German: "
        input_texts = [task_prefix + self.preprocess_requests(r) for r in requests]

        logger.debug("Received texts: '%s'", input_texts)
        inputs = self.tokenizer(
            input_texts,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        return inputs

    def preprocess_requests(self, request):
        """
        Preprocess request
        Args:
            request : Request to be decoded.
        Returns:
            str: Decoded input text
        """
        input_text = request.get("data") or request.get("body")
        if isinstance(input_text, (bytes, bytearray)):
            input_text = input_text.decode("utf-8")
        return input_text

    @torch.inference_mode()
    def inference(self, input_batch):
        """
        Generates the translated text for the given input
        Args:
            input_batch : A tensors: the batch of input ids and attention masks, as returned by the
            preprocess function.
        Returns:
            list: A list of strings with the translated text for each input text in the batch.
        """
        outputs = self.model.generate(
            **input_batch,
        )

        inferences = self.tokenizer.batch_decode(
            outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        logger.debug("Generated text: %s", inferences)
        return inferences

    def postprocess(self, inference_output):
        """Post Process Function converts the predicted response into Torchserve readable format.
        Args:
            inference_output (list): It contains the predicted response of the input text.
        Returns:
            (list): Returns a list of the Predictions.
        """
        return inference_output
