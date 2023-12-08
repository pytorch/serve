"""
Tokenizer from https://github.com/facebookresearch/llama/blob/main/llama/tokenizer.py.
"""
# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from logging import getLogger
from typing import List


logger = getLogger()


class Tokenizer:
    def __init__(self, model_path: str):
        # reload tokenizer
        from sentencepiece import SentencePieceProcessor

        self.sp_model = SentencePieceProcessor(model_file=model_path)  # pyre-ignore[28]
        logger.info(f"Reloaded SentencePiece model from {model_path}")

        # BOS / EOS token IDs
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()
        logger.info(
            f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}"
        )
        assert (
            self.sp_model.vocab_size()
            == self.sp_model.get_piece_size()  # pyre-ignore[16]
        )

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        t = self.sp_model.encode(s)  # pyre-ignore[16]
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        return self.sp_model.decode(t)  # pyre-ignore[16]