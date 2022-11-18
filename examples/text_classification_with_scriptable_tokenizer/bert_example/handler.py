"""
Module for text classification with scriptable tokenizer
"""
import json
import logging
from abc import ABC

import torch

# Necessary to successfully load the model (see https://github.com/pytorch/text/issues/1793)
import torchtext  # nopycln: import

from ts.torch_handler.base_handler import BaseHandler
from ts.utils.util import CLEANUP_REGEX

logger = logging.getLogger(__name__)


def remove_html_tags(text):
    """
    Removes html tags
    """
    clean_text = CLEANUP_REGEX.sub("", text)
    return clean_text


class CustomTextClassifier(BaseHandler, ABC):
    """
    TextClassifier handler class. This handler takes a text (string) and
    as input and returns the classification text based on the model vocabulary.
    Because the predefined TextHandler in ts/torch_handler defines unnecessary
    steps like loading a vocabulary file for the tokenizer, we define our handler
    starting from BaseHandler.
    """

    def preprocess(self, data):
        """
        Tokenization is dealt with inside the scripted model itself.
        We therefore only apply these basic cleanup operations :
            - remove html tags
            - lowercase all text

        Args:
            data (str): The input data is in the form of a string

        Returns:
            (Tensor): Text Tensor is returned after perfoming the pre-processing operations
            (str): The raw input is also returned in this function
        """

        # Compat layer: normally the envelope should just return the data
        # directly, but older versions of Torchserve didn't have envelope.
        # Processing only the first input, not handling batch inference
        batch_0, batch_1 = [], []

        for line in data:
            text = line.get("data") or line.get("body")
            # Decode text if not a str but bytes or bytearray
            if isinstance(text, (bytes, bytearray)):
                text = text.decode("utf-8")

            values = json.loads(text)

            assert "sequence_0" in values and "sequence_1" in values

            batch_0.append(remove_html_tags(values["sequence_0"]))
            batch_1.append(remove_html_tags(values["sequence_1"]))

        return batch_0, batch_1

    def inference(self, data, *args, **kwargs):
        """The Inference Request is made through this function and the user
        needs to override the inference function to customize it.

        Args:
            data (torch tensor): The data is in the form of Torch Tensor
                                 whose shape should match that of the
                                  Model Input shape.

        Returns:
            (Torch Tensor): The predicted response from the model is returned
                            in this function.
        """
        with torch.no_grad():
            results = self.model(*data)
        return results

    def _load_torchscript_model(self, model_pt_path):
        """Loads the PyTorch model and returns the NN model object.

        Args:
            model_pt_path (str): denotes the path of the model file.

        Returns:
            (NN Model Object) : Loads the model object.
        """
        # TODO: remove this method if https://github.com/pytorch/text/issues/1793 gets resolved
        model = torch.jit.load(model_pt_path)
        model.to(self.device)
        return model
