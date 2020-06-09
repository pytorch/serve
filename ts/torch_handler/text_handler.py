# pylint: disable=W0223
# Details : https://github.com/PyCQA/pylint/issues/3098
"""
Base module for all text based default handler.
Contains various text based utility methods
"""
import re
import string
import unicodedata
from abc import ABC
import torch
from torchtext.data.utils import get_tokenizer
from .base_handler import BaseHandler
from .contractions import CONTRACTION_MAP


class TextHandler(BaseHandler, ABC):
    """
    Base class for all text based default handler.
    Contains various text based utility methods
    """
    def __init__(self):
        super(TextHandler, self).__init__()
        self.source_vocab = None
        self.tokenizer = get_tokenizer('basic_english')

    def initialize(self, ctx):
        super(TextHandler, self).initialize(ctx)
        self.initialized = False
        self.source_vocab = torch.load(self.manifest['model']['sourceVocab'])
        self.initialized = True

    def _expand_contractions(self, text):
        def expand_match(contraction):
            match = contraction.group(0)
            first_char = match[0]
            expanded_contraction = CONTRACTION_MAP.get(match) if CONTRACTION_MAP.get(match) else CONTRACTION_MAP.get(
                match.lower())
            expanded_contraction = first_char + expanded_contraction[1:]
            return expanded_contraction

        contractions_pattern = re.compile('({})'.format('|'.join(CONTRACTION_MAP.keys())),
                                          flags=re.IGNORECASE | re.DOTALL)
        text = contractions_pattern.sub(expand_match, text)
        text = re.sub("'", "", text)
        return text

    def _remove_accented_characters(self, text):
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        return text

    def _remove_html_tags(self, text):
        cleanup_regex = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
        clean_text = re.sub(cleanup_regex, '', text)
        return clean_text

    def _remove_puncutation(self, text):
        return text.translate(str.maketrans('', '', string.punctuation))

    def _tokenize(self, text):
        return self.tokenizer(text)
