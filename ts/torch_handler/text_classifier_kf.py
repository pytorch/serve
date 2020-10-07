# pylint: disable=E1102
""" TODO remove pylint disable comment after
https://github.com/pytorch/pytorch/issues/24807 gets merged.
Module for text classification default handler
DOES NOT SUPPORT BATCH!
"""
import logging
from captum.attr import TokenReferenceBase
import torch
import torch.nn.functional as F
from torchtext.data.utils import ngrams_iterator
from ..utils.util import map_class_to_label
from .text_handler import TextHandler
logger = logging.getLogger(__name__)


class TextClassifier(TextHandler):
    """
    TextClassifier handler class. This handler takes a text (string) and
    as input and returns the classification text based on the model vocabulary.
    """

    ngrams = 2

    def preprocess(self, data):
        """
        Normalizes the input text for PyTorch model using following basic cleanup operations :
            - remove html tags
            - lowercase all text
            - expand contractions [like I'd -> I would, don't -> do not]
            - remove accented characters
            - remove punctuations
        Converts the normalized text to tensor using the source_vocab.
        Returns a Tensor
        """

        # Compat layer: normally the envelope should just return the data
        # directly, but older versions of Torchserve didn't have envelope.
        # Processing only the first input, not handling batch inference
        text = None
        row = data[0]
        if isinstance(row, dict):
            text = row.get("data") or row.get("body") or row
            logger.info("The text recieved in text_classifier %s", text)
        else:
            text = row
        # text = text.decode('utf-8')

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
            device=self.device,
        )
        return text_tensor, text

    def inference(self, data, *args, **kwargs):
        """
        Override to customize the inference
        :param data: Torch tensor, matching the model input shape
        :return: Prediction output as Torch tensor
        """
        text_tensor, _ = data
        offsets = torch.as_tensor([0], device=self.device)
        return super().inference(text_tensor, offsets)

    def postprocess(self, data):
        """
        Override to customize the post-processing
        :param data: Torch tensor, containing prediction output from the model
        :return: Python list
        """
        logger.info("inference shape %d", data.shape)
        data = F.softmax(data)
        data = data.tolist()
        return map_class_to_label(data, self.mapping)

    def get_insights(self, text_preprocess, _, target=0):
        """
        Calculates the captum insights
        :param text_tensor: The preprocessed tensor
        :param text: Unprocessed text to get word tokens
        :return: dict
        """
        text_tensor, all_tokens = text_preprocess
        token_reference = TokenReferenceBase()
        logger.info("input_text shape %d", len(text_tensor.shape))
        logger.info("get_insights target %d", target)
        offsets = torch.tensor([0])

        all_tokens = self.get_word_token(all_tokens)
        logger.info("text_tensor tokenized shape %d", text_tensor.shape)
        reference_indices = token_reference.generate_reference(
            text_tensor.shape[0], device=self.device
        ).squeeze(0)
        logger.info("reference indices shape %d", reference_indices.shape)

        # all_tokens = self.get_word_token(text)
        attributions = self.lig.attribute(
            text_tensor,
            reference_indices,
            additional_forward_args=(offsets),
            return_convergence_delta=False,
            target=target
        )

        logger.info("attributions shape %d",attributions.shape)
        attributions_sum = self.summarize_attributions(attributions)
        response = {}

        response["importances"] = attributions_sum.tolist()
        response["words"] = all_tokens
        return [response]
