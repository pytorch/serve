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
            [self.source_vocab[token] for token in ngrams_iterator(text, self.ngrams)],
            device=self.device,
        )
        return text_tensor

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
        offsets = torch.as_tensor([0], device=self.device)
        return super().inference(data, offsets)

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
        response = {}
        logger.info("inference shape %s", data.shape)
        data = F.softmax(data)
        data = data.tolist()
        response["predictions"] = map_class_to_label(data, self.mapping)

        return [response]
