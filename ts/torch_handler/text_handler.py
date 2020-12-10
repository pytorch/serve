"""
Base module for all text based default handler.
Contains various text based utility methods
"""
import os
import re
import logging
import string
import unicodedata
from abc import ABC
import torch
import torch.nn.functional as F
from torchtext.data.utils import get_tokenizer
from captum.attr import LayerIntegratedGradients
from .base_handler import BaseHandler
from .contractions import CONTRACTION_MAP

logger = logging.getLogger(__name__)


CLEANUP_REGEX = re.compile("<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});")
CONTRACTIONS_PATTERN = re.compile(
    "({})".format("|".join(CONTRACTION_MAP.keys())),
    flags=re.IGNORECASE | re.DOTALL,
)


class TextHandler(BaseHandler, ABC):
    """
    Base class for all text based default handler.
    Contains various text based utility methods
    """

    def __init__(self):
        super().__init__()
        self.source_vocab = None
        self.tokenizer = get_tokenizer("basic_english")
        self.input_text = None
        self.lig = None
        self.initialized = None

    def initialize(self, context):
        """
        Loads the model and Initializes the necessary artifacts
        """
        super().initialize(context)
        self.initialized = False
        source_vocab = self.manifest['model']['sourceVocab'] if 'sourceVocab' in self.manifest['model'] else None
        if source_vocab:
            # Backward compatibility
            self.source_vocab = torch.load(source_vocab)
        else:
            self.source_vocab = torch.load(self.get_source_vocab_path(context))
        #Captum initialization
        self.lig = LayerIntegratedGradients(self.model, self.model.embedding)
        self.initialized = True

    def get_source_vocab_path(self, ctx):
        properties = ctx.system_properties
        model_dir = properties.get("model_dir")
        source_vocab_path = os.path.join(model_dir, "source_vocab.pt")

        if os.path.isfile(source_vocab_path):
            return source_vocab_path
        else:
            raise Exception('Missing the source_vocab file. Refer default handler '
                            'documentation for details on using text_handler.')

    def _expand_contractions(self, text):
        """
        Expands the contracted words in the text
        """

        def expand_match(contraction):
            match = contraction.group(0)
            first_char = match[0]
            expanded_contraction = (
                CONTRACTION_MAP.get(match)
                if CONTRACTION_MAP.get(match)
                else CONTRACTION_MAP.get(match.lower())
            )
            expanded_contraction = first_char + expanded_contraction[1:]
            return expanded_contraction

        text = CONTRACTIONS_PATTERN.sub(expand_match, text)
        text = re.sub("'", "", text)
        return text

    def _remove_accented_characters(self, text):
        """
        Removes remove_accented_characters
        """
        text = (
            unicodedata.normalize("NFKD", text)
            .encode("ascii", "ignore")
            .decode("utf-8", "ignore")
        )
        return text

    def _remove_html_tags(self, text):
        """
        Removes html tags
        """
        clean_text = CLEANUP_REGEX.sub("", text)
        return clean_text

    def _remove_puncutation(self, *args, **kwargs):
        """
        Mispelled in original version. This is a compat layer
        """
        return self._remove_punctuation(*args, **kwargs)

    def _remove_punctuation(self, text):
        """
        Removes punctuation
        """
        return text.translate(str.maketrans("", "", string.punctuation))

    def _tokenize(self, text):
        return self.tokenizer(text)

    def get_word_token(self, input_tokens):
        """
        Constructs word tokens from text
        """
        # Remove unicode space character from BPE Tokeniser
        tokens = [token.replace("Ġ", "") for token in input_tokens]
        return tokens

    def summarize_attributions(self, attributions):
        """
        Summarises the attribution across multiple runs
        """
        attributions = F.softmax(attributions)
        attributions_sum = attributions.sum(dim=-1)
        logger.info("attributions sum shape %d", attributions_sum.shape)
        attributions = attributions / torch.norm(attributions_sum)
        return attributions
