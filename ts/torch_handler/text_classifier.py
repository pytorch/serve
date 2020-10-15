# pylint: disable=E1102
# TODO remove pylint disable comment after https://github.com/pytorch/pytorch/issues/24807 gets merged.
"""
Module for text classification default handler
DOES NOT SUPPORT BATCH!
"""
import torch
import torch.nn.functional as F
from torchtext.data.utils import ngrams_iterator
from .text_handler import TextHandler
from ..utils.util  import map_class_to_label

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

        row = data[0]
        # Compat layer: normally the envelope should just return the data
        # directly, but older versions of Torchserve didn't have envelope.
        if isinstance(row, dict):
            text = row.get("data") or row.get("body") or row
        else:
            text = row
        text = text.decode('utf-8')

        text = self._remove_html_tags(text)
        text = text.lower()
        text = self._expand_contractions(text)
        text = self._remove_accented_characters(text)
        text = self._remove_punctuation(text)
        text = self._tokenize(text)
        text = torch.as_tensor(
            [
                self.source_vocab[token]
                for token in ngrams_iterator(text, self.ngrams)
            ],
            device=self.device
        )
        return text

    def inference(self, data, *args, **kwargs):
        offsets = torch.as_tensor([0], device=self.device)
        return super().inference(data, offsets)

    def postprocess(self, data, output_explain = None):
        data = F.softmax(data)
        data = data.tolist()
        return map_class_to_label(data, self.mapping)
