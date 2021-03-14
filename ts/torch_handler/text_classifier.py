# pylint: disable=E1102
# TODO remove pylint disable comment after https://github.com/pytorch/pytorch/issues/24807 gets merged.
"""
Module for text classification default handler
DOES NOT SUPPORT BATCH!
"""
import logging
import torch
import torch.nn.functional as F
from torchtext.data.utils import ngrams_iterator
from captum.attr import TokenReferenceBase
from .text_handler import TextHandler
from ..utils.util import map_class_to_label

logger = logging.getLogger(__name__)

class TextClassifier(TextHandler):
    """
    TextClassifier handler class. This handler takes a text (string) and
    as input and returns the classification text based on the model vocabulary.
    """

    ngrams = 2

    def preprocess(self, data):
        """Normalizes the input text for PyTorch model using following basic cleanup operations :
            - remove html tags
            - lowercase all text
            - expand contractions [like I'd -> I would, don't -> do not]
            - remove accented characters
            - remove punctuations
        Converts the normalized text to tensor using the source_vocab.

        Args:
            data (str): The input data is in the form of a string

        Returns:
            (Tensor): Text Tensor is returned after perfoming the pre-processing operations
            (str): The raw input is also returned in this function
        """

        # Compat layer: normally the envelope should just return the data
        # directly, but older versions of Torchserve didn't have envelope.
        # Processing only the first input, not handling batch inference

        line = data[0]
        text = line.get("data") or line.get("body")
        if isinstance(text, (bytes, bytearray)):
            text = text.decode('utf-8')

        text = self._remove_html_tags(text)
        text = text.lower()
        text = self._expand_contractions(text)
        text = self._remove_accented_characters(text)
        text = self._remove_punctuation(text)
        text = self._tokenize(text)
        text_tensor = torch.as_tensor(
            [
                self.source_vocab[token]
                for token in ngrams_iterator(text, self.ngrams)
            ],
            device=self.device
        )
        return text_tensor, text

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
        text_tensor, _ = data
        offsets = torch.as_tensor([0], device=self.device)
        return super().inference(text_tensor, offsets)

    def postprocess(self, data):
        """
        The post process function converts the prediction response into a
           Torchserve compatible format

        Args:
            data (Torch Tensor): The data parameter comes from the prediction output
            output_explain (None): Defaults to None.

        Returns:
            (list): Returns the response containing the predictions and explanations
                    (if the Endpoint is hit).It takes the form of a list of dictionary.
        """
        data = F.softmax(data)
        data = data.tolist()
        return map_class_to_label(data, self.mapping)

    def get_insights(self, text_preprocess, _, target=0):
        """Calculates the captum insights

        Args:
            text_preprocess (tensor): Tensor of the Text Input
            _ (str): The Raw text data specified in the input request
            target (int): Defaults to 0, the user needs to specify the target
                          for the captum explanation.

        Returns:
            (dict): Returns a dictionary of the word token importances
        """
        text_tensor, all_tokens = text_preprocess
        token_reference = TokenReferenceBase()
        logger.info("input_text shape %s", len(text_tensor.shape))
        logger.info("get_insights target %s", target)
        offsets = torch.tensor([0]).to(self.device)

        all_tokens = self.get_word_token(all_tokens)
        logger.info("text_tensor tokenized shape %s", text_tensor.shape)
        reference_indices = token_reference.generate_reference(
            text_tensor.shape[0], device=self.device
        ).squeeze(0)
        logger.info("reference indices shape %s", reference_indices.shape)

        # all_tokens = self.get_word_token(text)
        attributions = self.lig.attribute(
            text_tensor,
            reference_indices,
            additional_forward_args=(offsets),
            return_convergence_delta=False,
            target=target,
        )

        logger.info("attributions shape %s", attributions.shape)
        attributions_sum = self.summarize_attributions(attributions)
        response = {}

        response["importances"] = attributions_sum.tolist()
        response["words"] = all_tokens
        return [response]
