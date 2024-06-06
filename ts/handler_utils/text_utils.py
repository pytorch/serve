"""
Functions which have been copied from TorchText to remove TorchServe's
dependency on TorchText

from torchtext.data.utils import ngrams_iterator, get_tokenizer

"""

import re


def ngrams_iterator(token_list, ngrams):
    """Return an iterator that yields the given tokens and their ngrams.

    Args:
        token_list: A list of tokens
        ngrams: the number of ngrams.

    Examples:
        >>> token_list = ['here', 'we', 'are']
        >>> list(ngrams_iterator(token_list, 2))
        >>> ['here', 'here we', 'we', 'we are', 'are']
    """

    def _get_ngrams(n):
        return zip(*[token_list[i:] for i in range(n)])

    for x in token_list:
        yield x
    for n in range(2, ngrams + 1):
        for x in _get_ngrams(n):
            yield " ".join(x)


_patterns = [
    r"\'",
    r"\"",
    r"\.",
    r"<br \/>",
    r",",
    r"\(",
    r"\)",
    r"\!",
    r"\?",
    r"\;",
    r"\:",
    r"\s+",
]

_replacements = [
    " '  ",
    "",
    " . ",
    " ",
    " , ",
    " ( ",
    " ) ",
    " ! ",
    " ? ",
    " ",
    " ",
    " ",
]

_patterns_dict = list((re.compile(p), r) for p, r in zip(_patterns, _replacements))


def _basic_english_normalize(line):
    r"""
    Basic normalization for a line of text.
    Normalization includes
    - lowercasing
    - complete some basic text normalization for English words as follows:
        add spaces before and after '\''
        remove '\"',
        add spaces before and after '.'
        replace '<br \/>'with single space
        add spaces before and after ','
        add spaces before and after '('
        add spaces before and after ')'
        add spaces before and after '!'
        add spaces before and after '?'
        replace ';' with single space
        replace ':' with single space
        replace multiple spaces with single space

    Returns a list of tokens after splitting on whitespace.
    """

    line = line.lower()
    for pattern_re, replaced_str in _patterns_dict:
        line = pattern_re.sub(replaced_str, line)
    return line.split()


def _split_tokenizer(x):  # noqa: F821
    return x.split()


def get_tokenizer(tokenizer, language="en"):
    r"""
    Generate tokenizer function for a string sentence.

    Args:
        tokenizer: the name of tokenizer function. If None, it returns split()
            function, which splits the string sentence by space.
            If basic_english, it returns _basic_english_normalize() function,
            which normalize the string first and split by space. If a callable
            function, it will return the function. If a tokenizer library
            (e.g. spacy, moses, toktok, revtok, subword), it returns the
            corresponding library.
        language: Default en

    Examples:
        >>> tokenizer = get_tokenizer("basic_english")
        >>> tokens = tokenizer("You can now install TorchText using pip!")
        >>> tokens
        >>> ['you', 'can', 'now', 'install', 'torchtext', 'using', 'pip', '!']

    """

    # default tokenizer is string.split(), added as a module function for serialization
    if tokenizer is None:
        return _split_tokenizer

    if tokenizer == "basic_english":
        if language != "en":
            raise ValueError("Basic normalization is only available for Enlish(en)")
        return _basic_english_normalize
