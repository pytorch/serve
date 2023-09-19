from typing import Optional

from transformers import AutoTokenizer
from transformers.generation.streamers import BaseStreamer, TextIteratorStreamer


class TextIteratorStreamerBatch(BaseStreamer):
    def __init__(
        self,
        tokenizer: "AutoTokenizer",
        batch_size: int,
        skip_prompt: bool = False,
        timeout: Optional[float] = None,
        **decode_kwargs,
    ):
        self.batch_size = batch_size
        self.streamers = [
            TextIteratorStreamer(tokenizer, skip_prompt, timeout, **decode_kwargs)
            for _ in range(batch_size)
        ]
        self.streamer_iterators = [iter(streamer) for streamer in self.streamers]

    def put(self, value):
        if value.shape[0] != self.batch_size:
            raise ValueError(
                f"TextIteratorStreamerBatch batch size is set to {self.batch_size} but got input tensor of shape {value.shape}"
            )

        for index in range(self.batch_size):
            self.streamers[index].put(value[index : index + 1])

    def end(self):
        for streamer in self.streamers:
            streamer.end()

    def __iter__(self):
        return self

    def __next__(self):
        values = []
        for iterator in self.streamer_iterators:
            try:
                values.append(next(iterator))
            except StopIteration:
                values.append(None)

        if None in values:
            raise StopIteration()

        return values
