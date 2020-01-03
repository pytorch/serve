from abc import ABC
from .base_handler import BaseHandler
from .contractions import CONTRACTION_MAP
import re
import spacy
import unicodedata
import torch


class TextHandler(BaseHandler, ABC):
    def __init__(self):
        super(TextHandler, self).__init__()
        self.source_vocab = None
        self.destination_vocab = None
        self.source_language = None
        self.spacy_model = None

    def initialize(self, ctx):
        super(TextHandler, self).initialize(ctx)
        self.initialized = False
        self.source_language = self.manifest['model']['sourceLanguage'] if 'sourceLanguage' in self.manifest['model'] else 'en'
        self.source_vocab = torch.load(self.manifest['model']['sourceVocab'])
        self.spacy_model = spacy.load(self.source_language)
        if 'destinationVocab' in self.manifest['model']:
            self.destination_vocab = torch.load(self.manifest['model']['destinationVocab'])
        self.initialized = True

    def _expand_contactions(self, text):
        def expand_match(contraction):
            match = contraction.group(0)
            first_char = match[0]
            expanded_contraction = CONTRACTION_MAP.get(match) if CONTRACTION_MAP.get(match) else CONTRACTION_MAP.get(
                match.lower())
            expanded_contraction = first_char + expanded_contraction[1:]
            return expanded_contraction

        if self.source_language == 'en':
            contractions_pattern = re.compile('({})'.format('|'.join(CONTRACTION_MAP.keys())),
                                              flags=re.IGNORECASE | re.DOTALL)
            text = contractions_pattern.sub(expand_match, text)
            text = re.sub("'", "", text)
        return text

    def _lemmatize_text(self, text):
        text = self._tokenize(text)
        text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])
        return text

    def _remove_accented_characters(self, text):
        if self.source_language == 'en':
            text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        return text

    def _remove_html_tags(self, text):
        cleanup_regex = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
        clean_text = re.sub(cleanup_regex, '', text)
        return clean_text

    def _remove_punctuations(self, text):
        tokens = self._tokenize(text)
        filtered_tokens = []
        for token in tokens:
            if not token.is_punct:
                filtered_tokens.append(token)
        return " ".join([tok.text for tok in filtered_tokens])

    def _tokenize(self, text):
        return self.spacy_model.tokenizer(text)
