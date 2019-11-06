import html
import logging
import os
import subprocess
from html.entities import html5, name2codepoint

import regex as re
from subword_nmt.apply_bpe import BPE


class Preprocessor(object):
    def __init__(self, bpe_code_file):
        super(Preprocessor, self).__init__()

        symbols = ''
        symbol_set = set({})

        for k in name2codepoint.keys():
            symbol_set.add(k)

        for k in html5.keys():
            symbol_set.add(k.strip(';'))

        for s in symbol_set:
            symbols += '|' + s

        symbols = symbols.strip('|')

        self.single = re.compile('&[ ]?(' + symbols + ')[ ]?;', re.IGNORECASE)
        self.double = re.compile('&[ ]?amp[ ]?;[ ]?(' + symbols + ')[ ]?;', re.IGNORECASE)

        self.singleNum = re.compile('&[ ]?#[ ]?([0-9]+)[ ]?;', re.IGNORECASE)
        self.doubleNum = re.compile('&[ ]?amp[ ]?;[ ]?#[ ]?([0-9]+)[ ]?;', re.IGNORECASE)

        self.singleXNum = re.compile('&[ ]?#[ ]?x[ ]?([a-f0-9]+)[ ]?;', re.IGNORECASE)
        self.doubleXNum = re.compile('&[ ]?amp[ ]?;[ ]?#[ ]?x[ ]?([a-f0-9]+)[ ]?;', re.IGNORECASE)

        self.nbsp = re.compile('(&[ ]?x?[ ]?n[]?b[ ]?([a-z][ ]?){0,6}[ ]?;)|(&[ ]?o[ ]?s[ ]?p[ ]?;)', re.IGNORECASE)

        self.shy = re.compile('[ ]?&[ ]?s[ ]?h[ ]?y[ ]?;[ ]?', re.IGNORECASE)

        self.bpe = None
        if bpe_code_file:
            with open(bpe_code_file, mode='r', encoding='utf-8') as f:
                self.bpe = BPE(f)
        else:
            logging.error('No BPE code file specified')

    def unescape(self, line):
        # put html-escaped (or double escaped) codes back into canonical format
        line = re.sub(self.double, r'&\1;', line)
        line = re.sub(self.doubleNum, r'&#\1;', line)
        line = re.sub(self.doubleXNum, r'&#x\1;', line)
        line = re.sub(self.single, r'&\1;', line)
        line = re.sub(self.singleNum, r'&#\1;', line)
        line = re.sub(self.singleXNum, r'&#x\1;', line)

        # get rid of this tag
        # alphabetic characters -- need only get rid of space around their canonical escaped forms
        line = re.sub(self.shy, '', line)

        # unescape
        line = html.unescape(line)

        # clean up weird errors in the escaping of the non-breaking space
        line = re.sub(self.nbsp, ' ', line)
        return line

    def bpe_encode(self, text):
        return self.bpe.process_line(text).strip()


class JoshuaPreprocessor(Preprocessor):
    def __init__(self, bpe_code_file, joshua_path, moses_path, lang):
        super(JoshuaPreprocessor, self).__init__(bpe_code_file)

        self.lang = lang
        self.normalizer = os.path.join(joshua_path, 'normalize.pl')
        self.tokenizer = os.path.join(moses_path, 'tokenizer.perl')
        self.cleaner = os.path.join(moses_path, 'remove-non-printing-char.perl')

        for f in [self.normalizer, self.tokenizer, self.cleaner]:
            os.chmod(f, 0o755)

    def run(self, text):
        text = self.unescape(text)

        # normalize, remove non-printing characters, and tokenize
        popen = subprocess.run(
            [self.normalizer, self.lang, '|', self.cleaner, '|', self.tokenizer, '-l', self.lang, '-no-escape', '-q'],
            input=text, encoding='utf-8', stdout=subprocess.PIPE)
        result = popen.stdout.strip()

        return self.bpe_encode(result)


class ChineseCharPreprocessor(JoshuaPreprocessor):
    def __init__(self, bpe_code_file, joshua_path, moses_path):
        super(ChineseCharPreprocessor, self).__init__(bpe_code_file, joshua_path, moses_path, 'zh')

        self.pattern = re.compile(
            '([\p{IsHan}\p{InCJK_Symbols_and_Punctuation}\p{InCJK_Radicals_Supplement}\p{InCJK_Compatibility}])',
            re.UNICODE)

    def run(self, text):
        text = self.unescape(text)

        # normalize and remove non-printing characters
        popen = subprocess.run([self.normalizer, self.lang, '|', self.cleaner], input=text, encoding='utf-8',
                               stdout=subprocess.PIPE)
        text = popen.stdout.strip()

        # tokenize by separating all ZH characters with a space
        text = self.pattern.sub(r' \1 ', text).strip()

        # tokenize other characters using Moses
        popen = subprocess.run([self.tokenizer, '-l', self.lang, '-no-escape', '-q'], input=text, encoding='utf-8',
                               stdout=subprocess.PIPE)
        result = popen.stdout.strip()

        return self.bpe_encode(result)


class Detokenizer():
    def __init__(self, path):
        self.de_bpe = re.compile('@@( |$)', re.IGNORECASE)
        self.de_tok = path

        os.chmod(self.de_tok, 0o755)

    def run(self, text):
        bpe_removed = re.sub(self.de_bpe, '', text.translation.strip())
        popen = subprocess.run([self.de_tok, '-l', 'en'], input=bpe_removed, encoding='utf-8', stdout=subprocess.PIPE,
                               env=os.environ)
        return popen.stdout.strip()
